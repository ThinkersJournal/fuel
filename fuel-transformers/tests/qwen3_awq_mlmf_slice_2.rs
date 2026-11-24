// SPDX-License-Identifier: MIT OR Apache-2.0
//! MLMF move, the AWQ read re-point (`docs/design-notes/2026-10-24-mlmf-move-survey.md`, finding F6):
//! `fuel_tensor::awq` builds an `AwqHostWeight` from `mlmf-awq`'s `AwqLayerBytes` instead of reading the
//! three tensors through its own `TensorView` reader, which is deleted.
//!
//! Equivalence, on the REAL `Orion-zhen/Qwen3-0.6B-AWQ` checkpoint (`FUEL_QWEN3_AWQ_DIR`; a `Hardware::Hub`
//! skip when absent): the kernel-layout host buffers (`weight`, `zeros`, `scales`, `zeros_w`) of EVERY
//! projection (28 layers x 7) hash to the digest the OLD reader produced, recorded below before it was
//! deleted. Positive control: 196 projections, more than a million words. Sabotage sibling: the digest
//! moves when one byte of one layer moves.

use std::path::PathBuf;

use fuel_test_support::hardware::{Hardware, Missing, skip};
use fuel_transformers::models::lazy_qwen3::Qwen3Config;

/// FNV-1a 64 over a byte stream.
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

fn checkpoint_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("FUEL_QWEN3_AWQ_DIR").map(PathBuf::from) else {
        skip(
            Hardware::Hub,
            Missing::device("FUEL_QWEN3_AWQ_DIR is not set (Orion-zhen/Qwen3-0.6B-AWQ, ~540 MB)"),
        );
        return None;
    };
    for f in ["config.json", "model.safetensors"] {
        if !dir.join(f).is_file() {
            skip(
                Hardware::Hub,
                Missing::device(format!("{} is missing {f}", dir.display())),
            );
            return None;
        }
    }
    Some(dir)
}

/// `(prefix, out_features, in_features)` of every projection the AWQ loader reads, in prefix order.
fn projections(cfg: &Qwen3Config) -> Vec<(String, usize, usize)> {
    let h = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let mut out = Vec::new();
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        for (name, (o, n)) in [
            ("self_attn.q_proj", (q_dim, h)),
            ("self_attn.k_proj", (kv_dim, h)),
            ("self_attn.v_proj", (kv_dim, h)),
            ("self_attn.o_proj", (h, q_dim)),
            ("mlp.gate_proj", (cfg.intermediate_size, h)),
            ("mlp.up_proj", (cfg.intermediate_size, h)),
            ("mlp.down_proj", (h, cfg.intermediate_size)),
        ] {
            out.push((format!("{p}.{name}"), o, n));
        }
    }
    out.sort();
    out
}

/// The digest of every projection's kernel-layout host buffers, read the way this test's `read` does.
fn digest_all<F>(dir: &std::path::Path, read: F) -> (u64, usize, usize)
where
    F: Fn(&str, usize, usize, usize) -> fuel_tensor::awq::AwqHostWeight,
{
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let v: serde_json::Value = serde_json::from_str(&config).expect("json");
    let group_size = v["quantization_config"]["group_size"]
        .as_u64()
        .expect("group_size") as usize;
    let cfg = Qwen3Config::from_hf_json_str(&config).expect("config");
    let mut h = FNV_BASIS;
    let (mut layers, mut words) = (0usize, 0usize);
    for (prefix, out, inp) in projections(&cfg) {
        let w = read(&prefix, inp, out, group_size);
        h = fnv(h, prefix.as_bytes());
        for x in w.weight.iter() {
            h = fnv(h, &x.to_le_bytes());
        }
        for x in w.zeros.iter() {
            h = fnv(h, &x.to_le_bytes());
        }
        for x in w.scales.iter() {
            h = fnv(h, &x.to_bits().to_le_bytes());
        }
        h = fnv(h, &(w.zeros_w as u64).to_le_bytes());
        layers += 1;
        words += w.weight.len() + w.zeros.len() + w.scales.len();
    }
    (h, layers, words)
}

/// The digest the OLD reader (`AwqHostWeight::load` over `MmapedSafetensors` `TensorView`s) produced on the
/// real checkpoint, recorded BEFORE that reader was deleted (196 projections, 58,920,960 words).
const OLD_READER_DIGEST: u64 = 0x37b8_7f80_1efc_3fb6;

/// Build every projection through the NEW path (`AwqFile` -> `AwqLayerBytes` -> `from_layer_bytes`) from
/// `bytes` and digest the host buffers.
fn digest_new_path(dir: &std::path::Path, bytes: &[u8]) -> (u64, usize, usize) {
    use fuel_loaders::awq::AwqFile;
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let v: serde_json::Value = serde_json::from_str(&config).expect("json");
    let group_size = v["quantization_config"]["group_size"]
        .as_u64()
        .expect("group_size") as usize;
    let file = AwqFile::open(bytes).expect("open");
    let layers: std::collections::BTreeMap<String, _> = file
        .locate(group_size)
        .expect("locate")
        .into_iter()
        .map(|l| (l.prefix.clone(), l))
        .collect();
    digest_all(dir, |prefix, ic, oc, g| {
        let layer = layers.get(prefix).expect("located");
        let lb = file.read_layer(layer).expect("read_layer");
        fuel_tensor::awq::AwqHostWeight::from_layer_bytes(&lb, ic, oc, g).expect("from_layer_bytes")
    })
}

#[test]
fn the_re_pointed_read_reproduces_the_old_readers_host_buffers_on_the_real_checkpoint() {
    let Some(dir) = checkpoint_dir() else { return };
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let (d, layers, words) = digest_new_path(&dir, &bytes);
    println!("NEW digest={d:#018x} layers={layers} words={words}");
    assert_eq!(layers, 196, "positive control: 28 layers x 7 projections");
    assert_eq!(words, 58_920_960);
    assert_eq!(d, OLD_READER_DIGEST);
}

/// Sabotage sibling: one byte of one layer's `qweight` moves the digest, so the equality above is a
/// comparison that can fail.
#[test]
fn one_flipped_qweight_byte_moves_the_digest() {
    use fuel_loaders::awq::AwqFile;
    let Some(dir) = checkpoint_dir() else { return };
    let pristine = std::fs::read(dir.join("model.safetensors")).expect("model.safetensors");
    let config = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let v: serde_json::Value = serde_json::from_str(&config).expect("json");
    let g = v["quantization_config"]["group_size"]
        .as_u64()
        .expect("group_size") as usize;
    let offset = {
        let file = AwqFile::open(&pristine).expect("open");
        let layers = file.locate(g).expect("locate");
        let q = file.read_layer(&layers[0]).expect("read_layer").qweight;
        q.bytes.as_ptr() as usize - pristine.as_ptr() as usize
    };
    let mut corrupted = pristine.clone();
    corrupted[offset] ^= 0xFF;
    let (d, _, _) = digest_new_path(&dir, &corrupted);
    assert_ne!(d, OLD_READER_DIGEST, "the sabotage must move the digest");
}
