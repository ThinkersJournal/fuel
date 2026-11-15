// SPDX-License-Identifier: MIT OR Apache-2.0
//! GGUF `qwen35` loader: [`Qwen3_5Config`] from the `qwen35.*` metadata and [`Qwen3_5Weights`]
//! from the tensors, K-quant weights kept quantized (ROADMAP row 118, milestone M1 of
//! `docs/design-notes/2026-11-14-qwen35-27b-plan.md`).
//!
//! # What is established and what is not
//!
//! The NAME map (`fuel_loaders::quantized::qwen35_names`) and the shapes are read off real files
//! (the 4B Q6_K blob, the 27B Q4_K_M header). The VALUE rules are **hypotheses until milestone M2**
//! compares logits against the CPU oracle. [`Qwen3_5GgufValueRules`] makes each one an explicit,
//! named choice, and [`Qwen3_5GgufReport::value_rules_verified`] is `false` until then. The default
//! follows the convention llama.cpp's converter documents for this family (the decoder norm gains are
//! stored final, i.e. already `1 + w`; `ssm_a` is `-exp(A_log)`); that is an answer key, not a
//! measurement here.
//!
//! # Memory
//!
//! A projection stays quantized when [`WeightStorage::from_ggml_bytes`] admits its type (Q8_0, Q4_K,
//! Q6_K). Every other type (the 27B file also carries IQ4_XS, Q5_K, IQ4_NL, IQ3_S and Q3_K, the
//! larger share of its tensors) is dequantized to f32, which is 4x to 8x the file's bytes. That
//! fallback is bounded by [`Qwen3_5GgufOptions::f32_fallback_budget_bytes`] and refused with an
//! error naming the tensor, never taken silently into an out-of-memory. Reaching those types
//! quantized needs `QuantType` variants and `QMatMul` kernels (ROADMAP row 155, M1 part b).
//! The token embedding is held as f32 (`Qwen3_5Weights::token_embedding`): 5 GB for the 27B.
//!
//! The multi-token-prediction block (`blk.<block_count - nextn_predict_layers>`, `nextn.*`) is
//! recorded in the report and not loaded: plain decode does not run it (not verified).

use super::lazy_qwen3_5::{
    FullAttentionWeights, GatedDeltaNetWeights, Qwen3_5Config, Qwen3_5Layer, Qwen3_5LayerKind,
    Qwen3_5Mixer, Qwen3_5Weights,
};
use fuel_ir::error::{Error, Result};
use fuel_loaders::quantized::GgmlDType;
use fuel_loaders::quantized::gguf_file::{Content, Value};
use fuel_loaders::quantized::gguf_mmap::MmapedContent;
use fuel_tensor::lazy::WeightStorage;
use std::path::Path;
use std::sync::Arc;

fn msg(text: String) -> Error {
    Error::Msg(format!("qwen35 gguf: {text}")).bt()
}

/// How GGUF tensor VALUES relate to the Hugging Face convention `Qwen3_5Weights` holds.
/// UNVERIFIED until milestone M2 (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3_5GgufValueRules {
    /// The decoder norm gains (`attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`,
    /// `output_norm`) are stored final (`1 + w` already). When `false` they are zero-centred and the
    /// loader adds `1`.
    pub norm_gain_is_final: bool,
    /// `ssm_a` holds `-exp(A_log)`. When `false` it holds `A_log` itself.
    pub ssm_a_is_neg_exp_of_a_log: bool,
}

impl Default for Qwen3_5GgufValueRules {
    fn default() -> Self {
        Self {
            norm_gain_is_final: true,
            ssm_a_is_neg_exp_of_a_log: true,
        }
    }
}

/// Loader options.
#[derive(Debug, Clone, Copy)]
pub struct Qwen3_5GgufOptions {
    /// The value rules to apply (hypotheses until M2).
    pub value_rules: Qwen3_5GgufValueRules,
    /// Most bytes of f32 the loader may allocate for projections whose type cannot stay quantized.
    /// Past it the load fails naming the tensor. Default 8 GiB.
    pub f32_fallback_budget_bytes: usize,
}

impl Default for Qwen3_5GgufOptions {
    fn default() -> Self {
        Self {
            value_rules: Qwen3_5GgufValueRules::default(),
            f32_fallback_budget_bytes: 8 << 30,
        }
    }
}

/// What the load did, so a caller can see what stayed quantized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3_5GgufReport {
    /// Projections kept quantized (`WeightStorage::Ggml`).
    pub quantized_projections: usize,
    /// Projections dequantized to f32 (name, the type they came in).
    pub dequantized_projections: Vec<(String, GgmlDType)>,
    /// Bytes the projections hold in host memory (`WeightStorage::resident_bytes`).
    pub projection_resident_bytes: usize,
    /// Blocks past the decoder (the multi-token-prediction head) that were not loaded.
    pub skipped_nextn_blocks: usize,
    /// `false` until milestone M2 proves the value rules against the oracle.
    pub value_rules_verified: bool,
}

/// A loaded model: its config, weights and the load report.
#[derive(Debug, Clone)]
pub struct Qwen3_5GgufLoad {
    pub config: Qwen3_5Config,
    pub weights: Qwen3_5Weights,
    pub report: Qwen3_5GgufReport,
}

fn get<'a>(content: &'a Content, key: &str) -> Result<&'a Value> {
    content
        .metadata
        .get(key)
        .ok_or_else(|| msg(format!("missing metadata key {key:?}")))
}

fn get_usize(content: &Content, key: &str) -> Result<usize> {
    Ok(get(content, key)?.to_u32()? as usize)
}

/// Build a [`Qwen3_5Config`] from `qwen35.*` metadata and the tensor table.
///
/// The layer kinds come from the tensors, not from `full_attention_interval`: a block with
/// `attn_q.weight` is full attention and one with `attn_qkv.weight` is a Gated DeltaNet layer; the
/// interval is then checked against that. Blocks past `block_count - nextn_predict_layers` are the
/// multi-token-prediction head and are not part of the decoder.
pub fn qwen3_5_config_from_gguf_content(content: &Content) -> Result<Qwen3_5Config> {
    let arch = get(content, "general.architecture")?.to_string()?;
    if arch != "qwen35" {
        return Err(msg(format!(
            "general.architecture is {arch:?}, expected \"qwen35\""
        )));
    }
    let block_count = get_usize(content, "qwen35.block_count")?;
    let nextn = content
        .metadata
        .get("qwen35.nextn_predict_layers")
        .map_or(Ok(0), |v| v.to_u32().map(|n| n as usize))?;
    let n_layers = block_count
        .checked_sub(nextn)
        .filter(|n| *n > 0)
        .ok_or_else(|| msg(format!("block_count {block_count} <= nextn {nextn}")))?;
    let interval = get_usize(content, "qwen35.full_attention_interval")?;
    let layer_types = (0..n_layers)
        .map(|i| {
            let has = |t: &str| content.tensor_infos.contains_key(&format!("blk.{i}.{t}"));
            match (has("attn_q.weight"), has("attn_qkv.weight")) {
                (true, false) => Ok(Qwen3_5LayerKind::FullAttention),
                (false, true) => Ok(Qwen3_5LayerKind::LinearAttention),
                _ => Err(msg(format!(
                    "blk.{i} has neither (or both of) attn_q.weight and attn_qkv.weight"
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    for (i, k) in layer_types.iter().enumerate() {
        let full = (i + 1) % interval == 0;
        if full != (*k == Qwen3_5LayerKind::FullAttention) {
            return Err(msg(format!(
                "blk.{i}: the tensors say {k:?}, full_attention_interval {interval} says {}",
                if full { "full attention" } else { "linear" }
            )));
        }
    }
    let head_dim = get_usize(content, "qwen35.attention.key_length")?;
    let heads = get_usize(content, "qwen35.attention.head_count")?;
    let hidden = get_usize(content, "qwen35.embedding_length")?;
    let embd = content
        .tensor_infos
        .get("token_embd.weight")
        .ok_or_else(|| content.missing_tensor_error("token_embd.weight"))?;
    let q = content
        .tensor_infos
        .get(
            &(0..n_layers)
                .find(|i| layer_types[*i] == Qwen3_5LayerKind::FullAttention)
                .map(|i| format!("blk.{i}.attn_q.weight"))
                .ok_or_else(|| msg("no full-attention block".to_string()))?,
        )
        .ok_or_else(|| msg("full-attention block lost its attn_q.weight".to_string()))?;
    let q_out = q.shape.dims()[0];
    let attn_output_gate = match q_out {
        n if n == 2 * heads * head_dim => true,
        n if n == heads * head_dim => false,
        n => {
            return Err(msg(format!(
                "attn_q.weight has {n} output rows; {heads} heads x {head_dim} gives {} (or {} gated)",
                heads * head_dim,
                2 * heads * head_dim
            )));
        }
    };
    let ssm_inner = get_usize(content, "qwen35.ssm.inner_size")?;
    let value_heads = get_usize(content, "qwen35.ssm.time_step_rank")?;
    if value_heads == 0 || !ssm_inner.is_multiple_of(value_heads) {
        return Err(msg(format!(
            "ssm.inner_size {ssm_inner} is not a multiple of ssm.time_step_rank {value_heads}"
        )));
    }
    let rope_dims = get_usize(content, "qwen35.rope.dimension_count")?;
    let cfg = Qwen3_5Config {
        vocab_size: embd.shape.dims()[0],
        hidden_size: hidden,
        intermediate_size: get_usize(content, "qwen35.feed_forward_length")?,
        num_hidden_layers: n_layers,
        layer_types,
        rms_norm_eps: get(content, "qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
        tie_word_embeddings: !content.tensor_infos.contains_key("output.weight"),
        max_position_embeddings: get_usize(content, "qwen35.context_length")?,
        num_attention_heads: heads,
        num_key_value_heads: get_usize(content, "qwen35.attention.head_count_kv")?,
        head_dim,
        attn_output_gate,
        rope_theta: get(content, "qwen35.rope.freq_base")?.to_f32()? as f64,
        partial_rotary_factor: rope_dims as f64 / head_dim as f64,
        linear_num_key_heads: get_usize(content, "qwen35.ssm.group_count")?,
        linear_num_value_heads: value_heads,
        linear_key_head_dim: get_usize(content, "qwen35.ssm.state_size")?,
        linear_value_head_dim: ssm_inner / value_heads,
        linear_conv_kernel_dim: get_usize(content, "qwen35.ssm.conv_kernel")?,
    };
    cfg.validate()?;
    Ok(cfg)
}

struct Reader<'a, 'b> {
    content: &'a Content,
    fetch: &'a dyn Fn(usize, usize) -> Result<&'b [u8]>,
    opts: Qwen3_5GgufOptions,
    quantized: usize,
    dequantized: Vec<(String, GgmlDType)>,
    resident: usize,
    f32_budget_left: usize,
}

impl<'b> Reader<'_, 'b> {
    fn tensor(&self, name: &str) -> Result<(&'b [u8], GgmlDType, Vec<usize>)> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .ok_or_else(|| self.content.missing_tensor_error(name))?;
        let elems = info.shape.elem_count();
        let len = elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let start = self.content.tensor_data_offset as usize + info.offset as usize;
        Ok((
            (self.fetch)(start, len)?,
            info.ggml_dtype,
            info.shape.dims().to_vec(),
        ))
    }

    fn f32s(&self, name: &str, want: usize) -> Result<Vec<f32>> {
        let (bytes, dt, dims) = self.tensor(name)?;
        let have: usize = dims.iter().product();
        if have != want {
            return Err(msg(format!(
                "tensor {name:?} has {have} elements ({dims:?}), the config implies {want}"
            )));
        }
        fuel_quantized::dequant_ggml_bytes(bytes, dt, name)
    }

    /// A zero-centred decoder norm gain, honouring the value rule.
    fn norm_gain(&self, name: &str, len: usize) -> Result<Arc<[f32]>> {
        let w = self.f32s(name, len)?;
        Ok(if self.opts.value_rules.norm_gain_is_final {
            w.into()
        } else {
            w.iter().map(|x| 1.0 + x).collect::<Vec<f32>>().into()
        })
    }

    /// A `[out, in]` projection: quantized when its type is admitted, else f32 within the budget.
    fn projection(&mut self, name: &str, out: usize, inp: usize) -> Result<WeightStorage> {
        let (bytes, dt, dims) = self.tensor(name)?;
        if dims != [out, inp] {
            return Err(msg(format!(
                "tensor {name:?} has shape {dims:?}, the config implies [{out}, {inp}]"
            )));
        }
        if let Some(w) = WeightStorage::from_ggml_bytes(dt, bytes, inp, out) {
            self.quantized += 1;
            self.resident += w.resident_bytes();
            return Ok(w);
        }
        let f32_bytes = out * inp * 4;
        if f32_bytes > self.f32_budget_left {
            return Err(msg(format!(
                "tensor {name:?} ({dt:?}) cannot stay quantized and needs {f32_bytes} bytes as f32, \
                 over the {} bytes left of the f32 fallback budget; its type has no QMatMul path yet",
                self.f32_budget_left
            )));
        }
        self.f32_budget_left -= f32_bytes;
        let row_major = fuel_quantized::dequant_ggml_bytes(bytes, dt, name)?;
        // [out, in] -> [in, out], the layout `WeightStorage::F32` holds.
        let mut t = vec![0.0_f32; out * inp];
        for o in 0..out {
            for i in 0..inp {
                t[i * out + o] = row_major[o * inp + i];
            }
        }
        self.dequantized.push((name.to_string(), dt));
        self.resident += f32_bytes;
        Ok(WeightStorage::F32(Arc::from(t)))
    }
}

impl Qwen3_5Weights {
    /// Load the decoder from a GGUF (single file or the shards of a `gguf-split` set).
    pub fn from_gguf_paths<P: AsRef<Path>>(
        paths: &[P],
        opts: Qwen3_5GgufOptions,
    ) -> Result<Qwen3_5GgufLoad> {
        let mc = MmapedContent::from_paths(paths)?;
        from_gguf_content_and_bytes(mc.content(), &|s, l| mc.bytes_at(s, l), opts)
    }
}

/// [`Qwen3_5Weights::from_gguf_paths`] over an already-parsed header and a byte fetcher
/// (`fetch(start, len)` is `len` bytes at absolute file position `start`).
pub fn from_gguf_content_and_bytes<'b>(
    content: &Content,
    fetch: &dyn Fn(usize, usize) -> Result<&'b [u8]>,
    opts: Qwen3_5GgufOptions,
) -> Result<Qwen3_5GgufLoad> {
    let cfg = qwen3_5_config_from_gguf_content(content)?;
    let mut r = Reader {
        content,
        fetch,
        opts,
        quantized: 0,
        dequantized: Vec::new(),
        resident: 0,
        f32_budget_left: opts.f32_fallback_budget_bytes,
    };
    let h = cfg.hidden_size;
    let token_embedding: Arc<[f32]> = r.f32s("token_embd.weight", cfg.vocab_size * h)?.into();
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        layers.push(load_layer(&mut r, &cfg, i)?);
    }
    let final_norm_gain = r.norm_gain("output_norm.weight", h)?;
    let output = if cfg.tie_word_embeddings {
        crate::models::lazy_llama_full::tied_lm_head_from_embeddings(
            &token_embedding,
            cfg.vocab_size,
            h,
        )
    } else {
        r.projection("output.weight", cfg.vocab_size, h)?
    };
    let nextn = content
        .metadata
        .get("qwen35.nextn_predict_layers")
        .map_or(Ok(0), |v| v.to_u32().map(|n| n as usize))?;
    let report = Qwen3_5GgufReport {
        quantized_projections: r.quantized,
        dequantized_projections: r.dequantized,
        projection_resident_bytes: r.resident,
        skipped_nextn_blocks: nextn,
        value_rules_verified: false,
    };
    Ok(Qwen3_5GgufLoad {
        weights: Qwen3_5Weights {
            instance: fuel_tensor::decode_shape::ModelInstanceId::next(),
            token_embedding,
            layers,
            final_norm_gain,
            output,
        },
        config: cfg,
        report,
    })
}

fn load_layer(r: &mut Reader<'_, '_>, cfg: &Qwen3_5Config, i: usize) -> Result<Qwen3_5Layer> {
    let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let p = format!("blk.{i}");
    let input_norm_gain = r.norm_gain(&format!("{p}.attn_norm.weight"), h)?;
    let post_attn_norm_gain = r.norm_gain(&format!("{p}.post_attention_norm.weight"), h)?;
    let mixer = match cfg.layer_types[i] {
        Qwen3_5LayerKind::LinearAttention => Qwen3_5Mixer::GatedDeltaNet(load_gdn(r, cfg, &p)?),
        Qwen3_5LayerKind::FullAttention => Qwen3_5Mixer::FullAttention(load_attn(r, cfg, &p)?),
    };
    Ok(Qwen3_5Layer {
        input_norm_gain,
        mixer,
        post_attn_norm_gain,
        ffn_gate: r.projection(&format!("{p}.ffn_gate.weight"), inter, h)?,
        ffn_up: r.projection(&format!("{p}.ffn_up.weight"), inter, h)?,
        ffn_down: r.projection(&format!("{p}.ffn_down.weight"), h, inter)?,
    })
}

fn load_gdn(r: &mut Reader<'_, '_>, cfg: &Qwen3_5Config, p: &str) -> Result<GatedDeltaNetWeights> {
    let h = cfg.hidden_size;
    let nv = cfg.linear_num_value_heads;
    let (conv_dim, z) = (cfg.linear_conv_dim(), cfg.linear_value_dim());
    let ssm_a = r.f32s(&format!("{p}.ssm_a"), nv)?;
    let a_log: Vec<f32> = if r.opts.value_rules.ssm_a_is_neg_exp_of_a_log {
        ssm_a
            .iter()
            .enumerate()
            .map(|(j, a)| {
                if *a < 0.0 {
                    Ok((-a).ln())
                } else {
                    Err(msg(format!(
                        "{p}.ssm_a[{j}] = {a}: the -exp(A_log) rule needs a negative value"
                    )))
                }
            })
            .collect::<Result<_>>()?
    } else {
        ssm_a
    };
    Ok(GatedDeltaNetWeights {
        in_proj_qkv: r.projection(&format!("{p}.attn_qkv.weight"), conv_dim, h)?,
        in_proj_z: r.projection(&format!("{p}.attn_gate.weight"), z, h)?,
        in_proj_a: r.projection(&format!("{p}.ssm_alpha.weight"), nv, h)?,
        in_proj_b: r.projection(&format!("{p}.ssm_beta.weight"), nv, h)?,
        conv1d_weight: r
            .f32s(
                &format!("{p}.ssm_conv1d.weight"),
                conv_dim * cfg.linear_conv_kernel_dim,
            )?
            .into(),
        a_log: a_log.into(),
        dt_bias: r.f32s(&format!("{p}.ssm_dt.bias"), nv)?.into(),
        norm_gain: r
            .f32s(&format!("{p}.ssm_norm.weight"), cfg.linear_value_head_dim)?
            .into(),
        out_proj: r.projection(&format!("{p}.ssm_out.weight"), h, z)?,
    })
}

fn load_attn(r: &mut Reader<'_, '_>, cfg: &Qwen3_5Config, p: &str) -> Result<FullAttentionWeights> {
    let h = cfg.hidden_size;
    let q_dim = cfg.attn_q_dim();
    let q_out = if cfg.attn_output_gate {
        2 * q_dim
    } else {
        q_dim
    };
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    Ok(FullAttentionWeights {
        q_proj: r.projection(&format!("{p}.attn_q.weight"), q_out, h)?,
        k_proj: r.projection(&format!("{p}.attn_k.weight"), kv, h)?,
        v_proj: r.projection(&format!("{p}.attn_v.weight"), kv, h)?,
        o_proj: r.projection(&format!("{p}.attn_output.weight"), h, q_dim)?,
        q_norm_gain: r.norm_gain(&format!("{p}.attn_q_norm.weight"), cfg.head_dim)?,
        k_norm_gain: r.norm_gain(&format!("{p}.attn_k_norm.weight"), cfg.head_dim)?,
    })
}
