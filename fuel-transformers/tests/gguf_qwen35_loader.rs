// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 118, milestone M1: the GGUF `qwen35` loader.
//!
//! Two kinds of evidence, kept apart. (1) A tiny synthetic `qwen35` file built here, loaded through
//! the real header parser: it shows what the loader DOES (config from metadata, which projections
//! stay quantized, the f32 fallback and its budget, each value rule applied as named). (2) The real
//! 27B Q4_K_M tensor table (`tests/fixtures/qwen35_27b_q4km_tensors.tsv`, header only): it shows the
//! config derives from that file's shapes and that all 866 tensors are accounted for. Neither shows
//! the VALUE rules are right; that is milestone M2 (`Qwen3_5GgufReport::value_rules_verified`).

use byteorder::{LittleEndian, WriteBytesExt};
use fuel_graph::QuantType;
use fuel_ir::quantized::GgmlDType;
use fuel_loaders::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
use fuel_loaders::quantized::qwen35_names::map_qwen35_gguf_name;
use fuel_quantized::{BlockQ6K, BlockQ8_0, GgmlType, blocks_as_bytes};
use fuel_tensor::lazy::WeightStorage;
use fuel_transformers::models::lazy_qwen3_5::{Qwen3_5LayerKind, Qwen3_5Mixer};
use fuel_transformers::models::lazy_qwen3_5_gguf::{
    Qwen3_5GgufOptions, Qwen3_5GgufValueRules, from_gguf_content_and_bytes,
    qwen3_5_config_from_gguf_content,
};
use std::collections::HashMap;
use std::path::Path;

const H: usize = 256;
const VOCAB: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 32;
const NK: usize = 2;
const NV: usize = 4;
const KEY_DIM: usize = 32;
const CONV_DIM: usize = 2 * NK * KEY_DIM + NV * KEY_DIM; // 256
const Z: usize = NV * KEY_DIM; // 128

// ---- a tiny GGUF writer (metadata + tensors) ---------------------------------------------------

struct T {
    name: String,
    dims: Vec<u64>,
    code: u32,
    data: Vec<u8>,
}

enum Kv {
    U32(&'static str, u32),
    F32(&'static str, f32),
    Str(&'static str, &'static str),
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    b.extend_from_slice(s.as_bytes());
}

fn build_gguf(kvs: &[Kv], tensors: &[T]) -> Vec<u8> {
    let align = |n: usize| n.div_ceil(32) * 32;
    let mut meta = Vec::new();
    for kv in kvs {
        match kv {
            Kv::U32(k, v) => {
                put_str(&mut meta, k);
                meta.write_u32::<LittleEndian>(4).unwrap();
                meta.write_u32::<LittleEndian>(*v).unwrap();
            }
            Kv::F32(k, v) => {
                put_str(&mut meta, k);
                meta.write_u32::<LittleEndian>(6).unwrap();
                meta.write_f32::<LittleEndian>(*v).unwrap();
            }
            Kv::Str(k, v) => {
                put_str(&mut meta, k);
                meta.write_u32::<LittleEndian>(8).unwrap();
                put_str(&mut meta, v);
            }
        }
    }
    let mut infos = Vec::new();
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for t in tensors {
        offsets.push(off);
        put_str(&mut infos, &t.name);
        infos
            .write_u32::<LittleEndian>(t.dims.len() as u32)
            .unwrap();
        for d in &t.dims {
            infos.write_u64::<LittleEndian>(*d).unwrap();
        }
        infos.write_u32::<LittleEndian>(t.code).unwrap();
        infos.write_u64::<LittleEndian>(off as u64).unwrap();
        off = align(off + t.data.len());
    }
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(0x4655_4747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
    buf.write_u64::<LittleEndian>(kvs.len() as u64).unwrap();
    buf.extend_from_slice(&meta);
    buf.extend_from_slice(&infos);
    buf.resize(align(buf.len()), 0);
    let base = buf.len();
    buf.resize(base + off, 0);
    for (t, o) in tensors.iter().zip(offsets) {
        buf[base + o..base + o + t.data.len()].copy_from_slice(&t.data);
    }
    buf
}

struct Lcg(u32);
impl Lcg {
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((self.0 >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.2
            })
            .collect()
    }
}

fn f32t(name: &str, dims: &[usize], v: &[f32]) -> T {
    T {
        name: name.into(),
        dims: dims.iter().rev().map(|d| *d as u64).collect(),
        code: 0,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn quant<B: GgmlType>(name: &str, out: usize, inp: usize, code: u32, rng: &mut Lcg) -> T {
    let w = rng.vec(out * inp);
    let mut blocks = vec![B::zeros(); w.len() / B::BLCK_SIZE];
    B::from_float(&w, &mut blocks);
    T {
        name: name.into(),
        dims: vec![inp as u64, out as u64],
        code,
        data: blocks_as_bytes(&blocks).to_vec(),
    }
}

/// Q8_0 (code 8) or Q6_K (code 14) projection `[out, in]`; F32 when `f32` is set.
fn proj(name: &str, out: usize, inp: usize, rng: &mut Lcg) -> T {
    if inp.is_multiple_of(256) {
        quant::<BlockQ6K>(name, out, inp, 14, rng)
    } else {
        quant::<BlockQ8_0>(name, out, inp, 8, rng)
    }
}

fn kvs(block_count: u32, interval: u32) -> Vec<Kv> {
    vec![
        Kv::Str("general.architecture", "qwen35"),
        Kv::U32("qwen35.block_count", block_count),
        Kv::U32("qwen35.nextn_predict_layers", 1),
        Kv::U32("qwen35.context_length", 4096),
        Kv::U32("qwen35.embedding_length", H as u32),
        Kv::U32("qwen35.feed_forward_length", 256),
        Kv::U32("qwen35.attention.head_count", HEADS as u32),
        Kv::U32("qwen35.attention.head_count_kv", KV_HEADS as u32),
        Kv::U32("qwen35.attention.key_length", HEAD_DIM as u32),
        Kv::F32("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
        Kv::F32("qwen35.rope.freq_base", 10_000_000.0),
        Kv::U32("qwen35.rope.dimension_count", 8),
        Kv::U32("qwen35.full_attention_interval", interval),
        Kv::U32("qwen35.ssm.conv_kernel", 4),
        Kv::U32("qwen35.ssm.state_size", KEY_DIM as u32),
        Kv::U32("qwen35.ssm.group_count", NK as u32),
        Kv::U32("qwen35.ssm.time_step_rank", NV as u32),
        Kv::U32("qwen35.ssm.inner_size", (NV * KEY_DIM) as u32),
    ]
}

/// Four decoder blocks (three Gated DeltaNet, then one full attention) plus a stub MTP block 4.
/// `ssm_a_value` fills `ssm_a`; `norm_value` fills every norm gain.
fn tensors(ssm_a_value: f32, norm_value: f32, f32_alpha: bool) -> Vec<T> {
    let mut rng = Lcg(7);
    let ones = |n: usize, v: f32| vec![v; n];
    let mut t = vec![
        f32t("token_embd.weight", &[VOCAB, H], &rng.vec(VOCAB * H)),
        f32t("output_norm.weight", &[H], &ones(H, norm_value)),
        proj("output.weight", VOCAB, H, &mut rng),
    ];
    for l in 0..4 {
        let p = format!("blk.{l}");
        t.push(f32t(
            &format!("{p}.attn_norm.weight"),
            &[H],
            &ones(H, norm_value),
        ));
        t.push(f32t(
            &format!("{p}.post_attention_norm.weight"),
            &[H],
            &ones(H, norm_value),
        ));
        t.push(proj(&format!("{p}.ffn_gate.weight"), 256, H, &mut rng));
        t.push(proj(&format!("{p}.ffn_up.weight"), 256, H, &mut rng));
        t.push(proj(&format!("{p}.ffn_down.weight"), H, 256, &mut rng));
        if l == 3 {
            t.push(proj(
                &format!("{p}.attn_q.weight"),
                2 * HEADS * HEAD_DIM,
                H,
                &mut rng,
            ));
            t.push(proj(
                &format!("{p}.attn_k.weight"),
                KV_HEADS * HEAD_DIM,
                H,
                &mut rng,
            ));
            t.push(proj(
                &format!("{p}.attn_v.weight"),
                KV_HEADS * HEAD_DIM,
                H,
                &mut rng,
            ));
            t.push(proj(
                &format!("{p}.attn_output.weight"),
                H,
                HEADS * HEAD_DIM,
                &mut rng,
            ));
            t.push(f32t(
                &format!("{p}.attn_q_norm.weight"),
                &[HEAD_DIM],
                &ones(HEAD_DIM, norm_value),
            ));
            t.push(f32t(
                &format!("{p}.attn_k_norm.weight"),
                &[HEAD_DIM],
                &ones(HEAD_DIM, norm_value),
            ));
        } else {
            t.push(proj(&format!("{p}.attn_qkv.weight"), CONV_DIM, H, &mut rng));
            t.push(proj(&format!("{p}.attn_gate.weight"), Z, H, &mut rng));
            let alpha = format!("{p}.ssm_alpha.weight");
            t.push(if f32_alpha {
                f32t(&alpha, &[NV, H], &rng.vec(NV * H))
            } else {
                proj(&alpha, NV, H, &mut rng)
            });
            t.push(proj(&format!("{p}.ssm_beta.weight"), NV, H, &mut rng));
            t.push(proj(&format!("{p}.ssm_out.weight"), H, Z, &mut rng));
            t.push(f32t(
                &format!("{p}.ssm_conv1d.weight"),
                &[CONV_DIM, 4],
                &rng.vec(CONV_DIM * 4),
            ));
            t.push(f32t(&format!("{p}.ssm_a"), &[NV], &ones(NV, ssm_a_value)));
            t.push(f32t(&format!("{p}.ssm_dt.bias"), &[NV], &rng.vec(NV)));
            t.push(f32t(
                &format!("{p}.ssm_norm.weight"),
                &[KEY_DIM],
                &ones(KEY_DIM, 0.95),
            ));
        }
    }
    // The multi-token-prediction block: present in the file, not part of the decoder.
    t.push(f32t("blk.4.nextn.enorm.weight", &[H], &ones(H, 1.0)));
    t
}

fn load(
    bytes: &[u8],
    opts: Qwen3_5GgufOptions,
) -> fuel_ir::Result<fuel_transformers::models::lazy_qwen3_5_gguf::Qwen3_5GgufLoad> {
    let content = Content::from_bytes(bytes, Path::new("synthetic-qwen35.gguf"))?;
    from_gguf_content_and_bytes(&content, &|s, l| Ok(&bytes[s..s + l]), opts)
}

fn file(ssm_a: f32, norm: f32, f32_alpha: bool) -> Vec<u8> {
    build_gguf(&kvs(5, 4), &tensors(ssm_a, norm, f32_alpha))
}

// ---- (1) what the loader does on a synthetic file ---------------------------------------------

#[test]
fn config_comes_from_the_metadata_and_the_tensor_kinds() {
    let bytes = file(-0.5, 1.0, false);
    let content = Content::from_bytes(&bytes, Path::new("c.gguf")).unwrap();
    let cfg = qwen3_5_config_from_gguf_content(&content).unwrap();
    assert_eq!(
        cfg.num_hidden_layers, 4,
        "block_count 5 minus one nextn block"
    );
    assert_eq!(
        cfg.layer_types,
        vec![
            Qwen3_5LayerKind::LinearAttention,
            Qwen3_5LayerKind::LinearAttention,
            Qwen3_5LayerKind::LinearAttention,
            Qwen3_5LayerKind::FullAttention
        ]
    );
    assert!(cfg.attn_output_gate, "attn_q has 2 x heads x head_dim rows");
    assert!(!cfg.tie_word_embeddings, "output.weight is present");
    assert_eq!((cfg.vocab_size, cfg.hidden_size), (VOCAB, H));
    assert_eq!(cfg.linear_value_head_dim, KEY_DIM);
    assert_eq!(cfg.linear_conv_dim(), CONV_DIM);
    assert!(
        (cfg.partial_rotary_factor - 0.25).abs() < 1e-12,
        "8 of 32 dims rotate"
    );
}

#[test]
fn k_quant_projections_stay_quantized_and_the_rest_is_reported() {
    let bytes = file(-0.5, 1.0, true);
    let l = load(&bytes, Qwen3_5GgufOptions::default()).unwrap();
    // The only f32-typed projection is the one the file stores as f32; it is reported by name.
    assert_eq!(
        l.report.dequantized_projections,
        vec![
            ("blk.0.ssm_alpha.weight".to_string(), GgmlDType::F32),
            ("blk.1.ssm_alpha.weight".to_string(), GgmlDType::F32),
            ("blk.2.ssm_alpha.weight".to_string(), GgmlDType::F32),
        ]
    );
    assert!(l.report.quantized_projections > 20, "{:?}", l.report);
    assert_eq!(l.report.skipped_nextn_blocks, 1);
    assert!(
        !l.report.value_rules_verified,
        "the value rules are hypotheses until M2"
    );
    // Q6_K for an input of 256, Q8_0 for the narrow ones.
    let Qwen3_5Mixer::GatedDeltaNet(g) = &l.weights.layers[0].mixer else {
        panic!("layer 0 is Gated DeltaNet");
    };
    let qt = |w: &WeightStorage| match w {
        WeightStorage::Ggml { qtype, .. } => Some(*qtype),
        _ => None,
    };
    assert_eq!(qt(&g.in_proj_qkv), Some(QuantType::Q6K));
    assert_eq!(qt(&g.out_proj), Some(QuantType::Q8_0));
    assert_eq!(
        qt(&g.in_proj_a),
        None,
        "the f32-typed tensor is held as f32"
    );
    // The instrument: resident bytes are far below an all-f32 load of the same projections.
    let all_f32: usize = 4 * 3 * 256 * 256 * 4; // ffn alone
    assert!(
        l.report.projection_resident_bytes < 4 * all_f32,
        "{:?}",
        l.report
    );
}

#[test]
fn the_f32_fallback_is_bounded_and_names_the_tensor() {
    let bytes = file(-0.5, 1.0, true);
    let opts = Qwen3_5GgufOptions {
        f32_fallback_budget_bytes: 16,
        ..Default::default()
    };
    let err = load(&bytes, opts).unwrap_err().to_string();
    assert!(err.contains("blk.0.ssm_alpha.weight"), "{err}");
    assert!(err.contains("budget"), "{err}");
}

#[test]
fn each_value_rule_is_applied_as_named() {
    let bytes = file(-0.5, 0.25, false);
    // Default: gains stored final (used as is), ssm_a = -exp(A_log).
    let l = load(&bytes, Qwen3_5GgufOptions::default()).unwrap();
    assert!((l.weights.final_norm_gain[0] - 0.25).abs() < 1e-7);
    let Qwen3_5Mixer::GatedDeltaNet(g) = &l.weights.layers[0].mixer else {
        panic!()
    };
    assert!(
        (g.a_log[0] - 0.5_f32.ln()).abs() < 1e-6,
        "a_log = ln(-ssm_a)"
    );
    assert!(
        (g.norm_gain[0] - 0.95).abs() < 1e-7,
        "the gated norm is never offset"
    );
    // The other readings: zero-centred gains get +1, ssm_a is A_log itself.
    let opts = Qwen3_5GgufOptions {
        value_rules: Qwen3_5GgufValueRules {
            norm_gain_is_final: false,
            ssm_a_is_neg_exp_of_a_log: false,
        },
        ..Default::default()
    };
    let l = load(&bytes, opts).unwrap();
    assert!((l.weights.final_norm_gain[0] - 1.25).abs() < 1e-7);
    let Qwen3_5Mixer::GatedDeltaNet(g) = &l.weights.layers[0].mixer else {
        panic!()
    };
    assert!((g.a_log[0] + 0.5).abs() < 1e-7);
    assert!((g.norm_gain[0] - 0.95).abs() < 1e-7);
}

#[test]
fn a_non_negative_ssm_a_refuses_the_neg_exp_rule() {
    let bytes = file(0.5, 1.0, false);
    let err = load(&bytes, Qwen3_5GgufOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("ssm_a"), "{err}");
}

#[test]
fn an_interval_that_contradicts_the_tensors_is_an_error() {
    // The tensors make block 3 full attention; an interval of 2 says block 1 is.
    let bytes = build_gguf(&kvs(5, 2), &tensors(-0.5, 1.0, false));
    let content = Content::from_bytes(&bytes, Path::new("i.gguf")).unwrap();
    let err = qwen3_5_config_from_gguf_content(&content)
        .unwrap_err()
        .to_string();
    assert!(err.contains("full_attention_interval"), "{err}");
}

#[test]
fn a_wrong_architecture_is_refused() {
    let mut k = kvs(5, 4);
    k[0] = Kv::Str("general.architecture", "qwen3");
    let bytes = build_gguf(&k, &tensors(-0.5, 1.0, false));
    let content = Content::from_bytes(&bytes, Path::new("a.gguf")).unwrap();
    assert!(qwen3_5_config_from_gguf_content(&content).is_err());
}

// ---- (2) the real 27B Q4_K_M tensor table --------------------------------------------------------

const TABLE: &str = include_str!("fixtures/qwen35_27b_q4km_tensors.tsv");

fn table() -> Vec<(String, Vec<usize>, u32)> {
    TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let mut c = l.split('\t');
            let name = c.next().unwrap().to_string();
            let raw: Vec<usize> = c
                .next()
                .unwrap()
                .split(',')
                .map(|d| d.parse().unwrap())
                .collect();
            let code: u32 = c.next().unwrap().parse().unwrap();
            (name, raw, code)
        })
        .collect()
}

fn content_27b() -> Content {
    let mut tensor_infos = HashMap::new();
    for (name, raw, code) in table() {
        // GGUF dims are innermost first; Fuel's shape is outermost first.
        let dims: Vec<usize> = raw.iter().rev().copied().collect();
        tensor_infos.insert(
            name,
            TensorInfo {
                ggml_dtype: GgmlDType::from_u32(code).unwrap(),
                shape: fuel_ir::Shape::from_dims(&dims),
                offset: 0,
            },
        );
    }
    let u = |v: u32| Value::U32(v);
    let metadata: HashMap<String, Value> = [
        ("general.architecture", Value::String("qwen35".into())),
        ("qwen35.block_count", u(65)),
        ("qwen35.nextn_predict_layers", u(1)),
        ("qwen35.context_length", u(262_144)),
        ("qwen35.embedding_length", u(5120)),
        ("qwen35.feed_forward_length", u(17_408)),
        ("qwen35.attention.head_count", u(24)),
        ("qwen35.attention.head_count_kv", u(4)),
        ("qwen35.attention.key_length", u(256)),
        ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("qwen35.rope.freq_base", Value::F32(10_000_000.0)),
        ("qwen35.rope.dimension_count", u(64)),
        ("qwen35.full_attention_interval", u(4)),
        ("qwen35.ssm.conv_kernel", u(4)),
        ("qwen35.ssm.state_size", u(128)),
        ("qwen35.ssm.group_count", u(16)),
        ("qwen35.ssm.time_step_rank", u(48)),
        ("qwen35.ssm.inner_size", u(6144)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    Content {
        magic: VersionedMagic::GgufV3,
        metadata,
        tensor_infos,
        unsupported_tensors: HashMap::new(),
        tensor_data_offset: 0,
    }
}

#[test]
fn the_27b_config_derives_from_the_real_tensor_table() {
    let rows = table();
    assert_eq!(rows.len(), 866, "the fixture is the whole table");
    let cfg = qwen3_5_config_from_gguf_content(&content_27b()).unwrap();
    assert_eq!(cfg.num_hidden_layers, 64, "65 blocks minus the MTP block");
    let full = cfg
        .layer_types
        .iter()
        .filter(|k| **k == Qwen3_5LayerKind::FullAttention)
        .count();
    assert_eq!((full, cfg.num_hidden_layers - full), (16, 48));
    assert_eq!((cfg.vocab_size, cfg.hidden_size), (248_320, 5120));
    assert!(cfg.attn_output_gate && !cfg.tie_word_embeddings);
    assert_eq!(cfg.linear_conv_dim(), 10_240);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert!((cfg.partial_rotary_factor - 0.25).abs() < 1e-12);
}

#[test]
fn all_866_tensors_of_the_27b_are_accounted_for() {
    let mut mapped = 0;
    let mut head = 0;
    let mut mtp = 0;
    let mut unaccounted = Vec::new();
    for (name, _, _) in table() {
        if map_qwen35_gguf_name(&name).is_some_and(|m| {
            // A decoder block index must be below 64.
            m.hf_name
                .strip_prefix("layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<usize>().ok())
                .is_none_or(|i| i < 64)
        }) {
            mapped += 1;
        } else if name == "output.weight" {
            head += 1;
        } else if name.starts_with("blk.64.") {
            mtp += 1;
        } else {
            unaccounted.push(name);
        }
    }
    assert!(unaccounted.is_empty(), "unaccounted: {unaccounted:?}");
    assert_eq!(head, 1);
    assert_eq!(
        mapped + head + mtp,
        866,
        "mapped {mapped}, head {head}, mtp {mtp}"
    );
    // Block 64 is the multi-token-prediction block: a full-attention block plus the `nextn.*`
    // tensors (15 tensors, read off the header).
    assert_eq!(mtp, 15, "the tensors of block 64");
}

#[test]
fn the_27b_matrices_all_stay_quantized_after_the_float_route_types() {
    // The M1 part (b) gap, closed (0.49.0): `WeightStorage::from_ggml_bytes` now admits Q8_0, Q4_K,
    // Q6_K plus Q3_K, Q5_K, IQ4_NL, IQ3_S and IQ4_XS, so no projection matrix of the real table
    // is left on the f32 path. The five extra types are asserted present, so this cannot pass
    // vacuously on a table that lost them.
    let admitted = |c: u32| matches!(c, 8 | 11 | 12 | 13 | 14 | 20 | 21 | 23);
    let mut stuck: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut seen: std::collections::BTreeMap<u32, usize> = Default::default();
    for (name, raw, code) in table() {
        let is_matrix = raw.len() == 2 && !name.contains("conv1d");
        if is_matrix && !name.starts_with("blk.64.") && code != 0 {
            *seen.entry(code).or_default() += 1;
            if !admitted(code) {
                *stuck.entry(code).or_default() += 1;
            }
        }
    }
    // Counts read off the header: Q3_K x2, Q5_K x142, IQ4_NL x9, IQ3_S x2, IQ4_XS x142.
    for (code, n) in [(11u32, 2usize), (13, 142), (20, 9), (21, 2), (23, 142)] {
        assert_eq!(seen.get(&code), Some(&n), "type {code}: {seen:?}");
    }
    assert!(
        stuck.is_empty(),
        "types still off the quantized path: {stuck:?}"
    );
}
