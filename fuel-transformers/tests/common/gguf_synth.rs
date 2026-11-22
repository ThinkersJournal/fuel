// SPDX-License-Identifier: MIT OR Apache-2.0
//! A hand-built GGUF of a one-layer llama, shared by `tests/gguf_resident_quantized.rs` and
//! `examples/gguf_resident_reach.rs` (ROADMAP row 124). Tensor infos only: `from_gguf` takes its
//! configuration from the caller, so no checkpoint is needed.

use byteorder::{LittleEndian, WriteBytesExt};
use fuel_quantized::{GgmlType, blocks_as_bytes};
use fuel_tensor::lazy::WeightStorage;
use fuel_transformers::models::lazy_llama_full::LlamaFullConfig;
use fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model;

pub const H: usize = 256;
pub const I: usize = 256;
pub const VOCAB: usize = 32;
pub const LAYERS: usize = 1;

pub fn cfg() -> LlamaFullConfig {
    LlamaFullConfig {
        hidden_size: H,
        intermediate_size: I,
        vocab_size: VOCAB,
        num_hidden_layers: LAYERS,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        tie_word_embeddings: false,
    }
}

pub struct Lcg(u32);
impl Lcg {
    pub fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((self.0 >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.2
            })
            .collect()
    }
}

/// One tensor to write: GGML type code, GGUF dims (innermost first), raw bytes.
pub struct T {
    pub name: String,
    pub dims: Vec<u64>,
    pub code: u32,
    pub data: Vec<u8>,
}

pub fn f32_tensor(name: &str, dims: &[u64], v: &[f32]) -> T {
    T {
        name: name.into(),
        dims: dims.to_vec(),
        code: 0,
        data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

/// Quantize `[out, in]` row-major values to `B`; return the block bytes and the values the
/// blocks decode to.
pub fn quantize<B: GgmlType>(w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let n_blocks = w.len() / B::BLCK_SIZE;
    let mut blocks = vec![B::zeros(); n_blocks];
    B::from_float(w, &mut blocks);
    let mut back = vec![0.0_f32; w.len()];
    B::to_float(&blocks, &mut back);
    (blocks_as_bytes(&blocks).to_vec(), back)
}

pub fn build_gguf(tensors: &[T]) -> Vec<u8> {
    let align = |n: usize| n.div_ceil(32) * 32;
    let mut infos = Vec::new();
    let mut offsets = Vec::new();
    let mut off = 0usize;
    for t in tensors {
        offsets.push(off);
        infos
            .write_u64::<LittleEndian>(t.name.len() as u64)
            .unwrap();
        infos.extend_from_slice(t.name.as_bytes());
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
    buf.write_u32::<LittleEndian>(0x4655_4747).unwrap(); // b"GGUF"
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // no metadata
    buf.extend_from_slice(&infos);
    buf.resize(align(buf.len()), 0);
    let base = buf.len();
    buf.resize(base + off, 0);
    for (t, o) in tensors.iter().zip(offsets) {
        buf[base + o..base + o + t.data.len()].copy_from_slice(&t.data);
    }
    buf
}

/// Every projection weight of the model, `(name, out, in)`.
pub fn projections() -> Vec<(String, usize, usize)> {
    let mut v = Vec::new();
    for l in 0..LAYERS {
        for (n, o, i) in [
            ("attn_q", H, H),
            ("attn_k", H, H),
            ("attn_v", H, H),
            ("attn_output", H, H),
            ("ffn_gate", I, H),
            ("ffn_up", I, H),
            ("ffn_down", H, I),
        ] {
            v.push((format!("blk.{l}.{n}.weight"), o, i));
        }
    }
    v.push(("output.weight".into(), VOCAB, H));
    v
}

/// Returns `(quantized file, dequantized-f32 file, quantized bytes of the projections)`.
pub fn files<B: GgmlType>(code: u32, plain_gain: f32) -> (Vec<u8>, Vec<u8>, usize) {
    let mut rng = Lcg(0x5eed);
    let mut quant = Vec::new();
    let mut plain = Vec::new();
    let mut qbytes = 0;
    let both = |q: &mut Vec<T>, p: &mut Vec<T>, name: &str, dims: &[u64], v: Vec<f32>| {
        q.push(f32_tensor(name, dims, &v));
        p.push(f32_tensor(name, dims, &v));
    };
    both(
        &mut quant,
        &mut plain,
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        rng.vec(H * VOCAB),
    );
    for l in 0..LAYERS {
        for n in ["attn_norm", "ffn_norm"] {
            both(
                &mut quant,
                &mut plain,
                &format!("blk.{l}.{n}.weight"),
                &[H as u64],
                vec![1.0; H],
            );
        }
    }
    both(
        &mut quant,
        &mut plain,
        "output_norm.weight",
        &[H as u64],
        vec![1.0; H],
    );
    for (name, o, i) in projections() {
        let w = rng.vec(o * i);
        let (bytes, back) = quantize::<B>(&w);
        qbytes += bytes.len();
        quant.push(T {
            name: name.clone(),
            dims: vec![i as u64, o as u64],
            code,
            data: bytes,
        });
        let back: Vec<f32> = back.iter().map(|v| v * plain_gain).collect();
        plain.push(f32_tensor(&name, &[i as u64, o as u64], &back));
    }
    (build_gguf(&quant), build_gguf(&plain), qbytes)
}

pub fn load(bytes: &[u8], tag: &str) -> QuantizedLlama3Model {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(format!("{tag}.gguf"));
    std::fs::write(&path, bytes).unwrap();
    QuantizedLlama3Model::from_gguf(&path, &cfg()).unwrap()
}

pub fn projection_weights(m: &QuantizedLlama3Model) -> Vec<&WeightStorage> {
    let w = &m.inner().inner.weights;
    let mut v = Vec::new();
    for l in &w.layers {
        v.extend([
            &l.attn_q,
            &l.attn_k,
            &l.attn_v,
            &l.attn_o,
            &l.ffn_gate,
            &l.ffn_up,
            &l.ffn_down,
        ]);
    }
    v.push(&w.output);
    v
}
