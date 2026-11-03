// SPDX-License-Identifier: MIT OR Apache-2.0
//! GGUF `qwen35` tensor names -> the Hugging Face names `Qwen3_5Weights` reads (ROADMAP row 118,
//! step two).
//!
//! **Status: the NAME map is read off a real file; the VALUE rules are hypotheses.** The GGUF names
//! and shapes come from `docs/design-notes/2026-11-01-qwen35-gguf-header.md` (the installed
//! `unsloth/Qwen3.5-4B-GGUF` Q6_K blob, header only). Which GGUF tensor corresponds to which HF
//! tensor is inferred from names and shapes, unchecked against HF weights. Two value
//! transforms are also unknown: whether the GGUF norm gains already include the `+1` of the
//! zero-centred convention, and whether `ssm_a` holds `A_log` or `-exp(A_log)`. They are returned
//! as `Unverified*` rules so a caller cannot mistake them for established fact. The CPU Hugging Face
//! oracle run (PM-scheduled, RAM) is what turns them into verified rules.
//!
//! Returned names are RELATIVE to the text tower prefix (`model.language_model.` in the multimodal
//! wrapper, `model.` in a text-only export), the same convention as `Qwen3_5Weights`.
//!
//! Not covered here: the q/k/v split inside `attn_qkv`, the gate-half order inside `attn_q`
//! (8192 = 2 x 16 x 256), and the GGUF `[in, out]` versus HF `[out, in]` shape flip. This module
//! maps names only.

/// How a GGUF tensor's values relate to the HF tensor. Everything but `AsIs` is a HYPOTHESIS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen35ValueRule {
    /// Same values as the HF tensor (up to the layout flip every matrix has).
    AsIs,
    /// UNVERIFIED: a decoder RMS-norm gain that may already include the zero-centred `+1`
    /// (`gguf = hf + 1`). `Qwen3_5Weights` stores `1 + w`, so if this holds the GGUF value is
    /// already the stored gain and must not get another `+1`.
    UnverifiedNormGainMayIncludePlusOne,
    /// UNVERIFIED: `ssm_a` may hold `-exp(A_log)` (llama.cpp's precomputed decay) rather than
    /// `A_log` itself.
    UnverifiedSsmAMayBeNegExpOfALog,
}

impl Qwen35ValueRule {
    /// `true` only for rules checked against an HF oracle. Today that is `AsIs` alone.
    pub const fn is_verified(self) -> bool {
        matches!(self, Qwen35ValueRule::AsIs)
    }
}

/// One mapped tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35Mapping {
    /// HF name relative to the text tower prefix, e.g. `layers.3.self_attn.q_proj.weight`.
    pub hf_name: String,
    pub rule: Qwen35ValueRule,
}

use Qwen35ValueRule::UnverifiedSsmAMayBeNegExpOfALog as NegExpALog;
use Qwen35ValueRule::{AsIs, UnverifiedNormGainMayIncludePlusOne as NormPlusOne};

/// Names outside the repeating blocks: (GGUF name, HF name, rule).
const GLOBAL_TENSORS: &[(&str, &str, Qwen35ValueRule)] = &[
    ("token_embd.weight", "embed_tokens.weight", AsIs),
    ("output_norm.weight", "norm.weight", NormPlusOne),
];

/// Names inside `blk.N.`: (GGUF tail, HF tail under `layers.N.`, rule). The DeltaNet pairings are
/// HYPOTHESES from shapes (see the module docs).
const BLOCK_TENSORS: &[(&str, &str, Qwen35ValueRule)] = &[
    ("attn_norm.weight", "input_layernorm.weight", NormPlusOne),
    (
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        NormPlusOne,
    ),
    ("ffn_gate.weight", "mlp.gate_proj.weight", AsIs),
    ("ffn_up.weight", "mlp.up_proj.weight", AsIs),
    ("ffn_down.weight", "mlp.down_proj.weight", AsIs),
    // Full-attention layers.
    ("attn_q.weight", "self_attn.q_proj.weight", AsIs),
    ("attn_k.weight", "self_attn.k_proj.weight", AsIs),
    ("attn_v.weight", "self_attn.v_proj.weight", AsIs),
    ("attn_output.weight", "self_attn.o_proj.weight", AsIs),
    ("attn_q_norm.weight", "self_attn.q_norm.weight", NormPlusOne),
    ("attn_k_norm.weight", "self_attn.k_norm.weight", NormPlusOne),
    // Gated DeltaNet layers.
    ("attn_qkv.weight", "linear_attn.in_proj_qkv.weight", AsIs),
    ("attn_gate.weight", "linear_attn.in_proj_z.weight", AsIs),
    ("ssm_alpha.weight", "linear_attn.in_proj_a.weight", AsIs),
    ("ssm_beta.weight", "linear_attn.in_proj_b.weight", AsIs),
    ("ssm_out.weight", "linear_attn.out_proj.weight", AsIs),
    ("ssm_conv1d.weight", "linear_attn.conv1d.weight", AsIs),
    ("ssm_a", "linear_attn.A_log", NegExpALog),
    ("ssm_dt.bias", "linear_attn.dt_bias", AsIs),
    // The gated norm inside a DeltaNet layer is NOT zero-centred in the HF checkpoint.
    ("ssm_norm.weight", "linear_attn.norm.weight", AsIs),
];

fn lookup(
    table: &[(&'static str, &'static str, Qwen35ValueRule)],
    key: &str,
) -> Option<(&'static str, Qwen35ValueRule)> {
    table
        .iter()
        .find(|(g, _, _)| *g == key)
        .map(|(_, hf, rule)| (*hf, *rule))
}

/// Map one GGUF tensor name of a `qwen35` file. `None` means the name is not part of the text
/// decoder the loader reads (for example `output.weight`, which this arch ties to the embedding,
/// or anything unrecognised): the caller decides whether that is an error.
pub fn map_qwen35_gguf_name(gguf: &str) -> Option<Qwen35Mapping> {
    if let Some((hf, rule)) = lookup(GLOBAL_TENSORS, gguf) {
        return Some(Qwen35Mapping {
            hf_name: hf.to_string(),
            rule,
        });
    }
    let (idx, tail) = gguf.strip_prefix("blk.")?.split_once('.')?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (hf, rule) = lookup(BLOCK_TENSORS, tail)?;
    Some(Qwen35Mapping {
        hf_name: format!("layers.{idx}.{hf}"),
        rule,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The tensor names of the installed 4B file (header note), for 32 blocks with full attention
    /// every fourth layer (3, 7, ..., 31).
    fn header_note_tensor_names() -> Vec<String> {
        let mut v = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
        ];
        for n in 0..32 {
            for t in [
                "attn_norm.weight",
                "post_attention_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                v.push(format!("blk.{n}.{t}"));
            }
            let full = [
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
            ];
            let delta = [
                "attn_qkv.weight",
                "attn_gate.weight",
                "ssm_alpha.weight",
                "ssm_beta.weight",
                "ssm_out.weight",
                "ssm_conv1d.weight",
                "ssm_a",
                "ssm_dt.bias",
                "ssm_norm.weight",
            ];
            let set: &[&str] = if n % 4 == 3 { &full } else { &delta };
            v.extend(set.iter().map(|t| format!("blk.{n}.{t}")));
        }
        v
    }

    #[test]
    fn header_note_population_is_426_and_every_name_maps() {
        let names = header_note_tensor_names();
        assert_eq!(
            names.len(),
            426,
            "fixture must reproduce the note's tensor count"
        );
        let unmapped: Vec<_> = names
            .iter()
            .filter(|n| map_qwen35_gguf_name(n).is_none())
            .collect();
        assert!(unmapped.is_empty(), "unmapped: {unmapped:?}");
    }

    #[test]
    fn map_is_injective_over_the_header_note_population() {
        let names = header_note_tensor_names();
        let hf: HashSet<String> = names
            .iter()
            .map(|n| map_qwen35_gguf_name(n).unwrap().hf_name)
            .collect();
        assert_eq!(
            hf.len(),
            names.len(),
            "two GGUF names mapped to one HF name"
        );
    }

    #[test]
    fn spot_checks_name_the_exact_hf_tensor() {
        let get = |n: &str| map_qwen35_gguf_name(n).unwrap();
        assert_eq!(get("token_embd.weight").hf_name, "embed_tokens.weight");
        assert_eq!(get("output_norm.weight").hf_name, "norm.weight");
        assert_eq!(
            get("blk.3.attn_q.weight").hf_name,
            "layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            get("blk.3.attn_output.weight").hf_name,
            "layers.3.self_attn.o_proj.weight"
        );
        assert_eq!(
            get("blk.10.attn_qkv.weight").hf_name,
            "layers.10.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(
            get("blk.10.attn_gate.weight").hf_name,
            "layers.10.linear_attn.in_proj_z.weight"
        );
        assert_eq!(get("blk.0.ssm_a").hf_name, "layers.0.linear_attn.A_log");
        assert_eq!(
            get("blk.0.ssm_dt.bias").hf_name,
            "layers.0.linear_attn.dt_bias"
        );
        assert_eq!(
            get("blk.31.post_attention_norm.weight").hf_name,
            "layers.31.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn hypothesis_rules_are_never_reported_verified() {
        let rule = |n: &str| map_qwen35_gguf_name(n).unwrap().rule;
        for n in [
            "blk.0.attn_norm.weight",
            "output_norm.weight",
            "blk.3.attn_q_norm.weight",
            "blk.0.ssm_a",
        ] {
            assert!(
                !rule(n).is_verified(),
                "{n} must stay unverified until the oracle run"
            );
        }
        // The DeltaNet gated norm is not zero-centred (HF mean 0.95): no +1 rule.
        assert_eq!(rule("blk.0.ssm_norm.weight"), Qwen35ValueRule::AsIs);
        assert!(rule("blk.0.ffn_up.weight").is_verified());
    }

    #[test]
    fn foreign_and_malformed_names_decline() {
        for n in [
            "output.weight",
            "blk.x.attn_q.weight",
            "blk..attn_q.weight",
            "blk.3",
            "blk.3.attn_q.bias",
            "blk.3.ffn_gate_exps.weight",
            "layers.3.self_attn.q_proj.weight",
            "",
        ] {
            assert!(map_qwen35_gguf_name(n).is_none(), "{n:?} should not map");
        }
    }
}
