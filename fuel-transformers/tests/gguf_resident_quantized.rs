// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 124: a Q8_0, Q4_K or Q6_K tensor in a GGUF stays quantized from load to the
//! matmul instead of being expanded to f32.
//!
//! The file is built by hand (tensor infos only: `from_gguf` takes its configuration from the
//! caller), so no real checkpoint is needed. Two files carry the same weights: one stores the
//! projections in the quantized format, the other stores their dequantized values as F32.
//! Loading the second is the old policy's outcome, so equal logits mean the quantized path
//! computes the same function while holding a fraction of the bytes.

use fuel_graph::QuantType;
use fuel_quantized::{BlockQ4K, BlockQ6K, BlockQ8_0, GgmlType};
use fuel_tensor::lazy::WeightStorage;

#[path = "common/gguf_synth.rs"]
mod synth;
use synth::{files, load, projection_weights, projections};

/// Largest logit difference between the quantized-resident model and the f32 model holding the
/// dequantized values times `plain_gain`, as a fraction of the largest f32 logit. Asserts the
/// residency of the quantized model on the way.
fn run<B: GgmlType>(code: u32, qtype: QuantType, plain_gain: f32) -> f32 {
    let (qfile, pfile, qbytes) = files::<B>(code, plain_gain);
    let q = load(&qfile, "quant");
    let p = load(&pfile, "plain");

    for (i, w) in projection_weights(&q).into_iter().enumerate() {
        assert!(
            matches!(w, WeightStorage::Ggml { qtype: t, .. } if *t == qtype),
            "projection {i} of the {qtype:?} file is not resident as Ggml({qtype:?})"
        );
    }
    let resident: usize = projection_weights(&q)
        .iter()
        .map(|w| w.resident_bytes())
        .sum();
    let f32_bytes: usize = projections().iter().map(|(_, o, i)| o * i * 4).sum();
    assert_eq!(
        resident, qbytes,
        "{qtype:?}: resident bytes must equal the block bytes in the file"
    );
    assert!(
        resident * 3 < f32_bytes,
        "{qtype:?}: {resident} resident bytes is not under a third of the {f32_bytes} f32 bytes"
    );
    // The control: the same weights stored as F32 are resident at 4 bytes per element, so the
    // instrument can read a dequantized model as one.
    let plain_resident: usize = projection_weights(&p)
        .iter()
        .map(|w| w.resident_bytes())
        .sum();
    assert_eq!(plain_resident, f32_bytes);

    let toks = [1_u32, 5, 9];
    let lq = q.forward(&toks, 0).unwrap().realize_f32();
    let lp = p.forward(&toks, 0).unwrap().realize_f32();
    assert_eq!(lq.len(), lp.len());
    let mut worst = 0.0_f32;
    let mut scale = 0.0_f32;
    for (a, b) in lq.iter().zip(&lp) {
        assert!(a.is_finite() && b.is_finite());
        worst = worst.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    eprintln!("{qtype:?} gain {plain_gain}: worst |dlogit| {worst:e} vs max |logit| {scale:e}");
    worst / scale.max(1e-3)
}

/// Measured on this box 2026-10-30 at gain 1.0: Q8_0 0.75%, Q4_K 1.1%, Q6_K 1.7% of the largest
/// logit (the CPU kernel quantizes activations to 8 bits, ROADMAP row 115, so it is not exact).
/// The bound sits above that and below what a 10% weight-scale error produces
/// (`a_wrong_weight_scale_is_caught`).
const LOGIT_TOL: f32 = 0.04;

fn check<B: GgmlType>(code: u32, qtype: QuantType) {
    let rel = run::<B>(code, qtype, 1.0);
    assert!(
        rel <= LOGIT_TOL,
        "{qtype:?}: logits differ from the dequantized-f32 model by {rel:e} of the largest logit"
    );
}

/// Sabotage sibling: the reference holds weights 10% too large, which the comparison must see.
fn catches<B: GgmlType>(code: u32, qtype: QuantType) {
    let rel = run::<B>(code, qtype, 1.10);
    assert!(
        rel > LOGIT_TOL,
        "{qtype:?}: a 10% weight-scale error moved the logits by only {rel:e}; LOGIT_TOL is too loose"
    );
}

#[test]
fn q8_0_stays_resident() {
    check::<BlockQ8_0>(8, QuantType::Q8_0);
}

#[test]
fn q4_k_stays_resident() {
    check::<BlockQ4K>(12, QuantType::Q4_K_M);
}

#[test]
fn q6_k_stays_resident() {
    check::<BlockQ6K>(14, QuantType::Q6K);
}

#[test]
fn a_wrong_weight_scale_is_caught() {
    catches::<BlockQ8_0>(8, QuantType::Q8_0);
    catches::<BlockQ4K>(12, QuantType::Q4_K_M);
    catches::<BlockQ6K>(14, QuantType::Q6K);
}

/// The loader keeps a tensor quantized only when it can: the refusals must be `None` (the
/// caller then dequantizes), never a half-formed storage.
#[test]
fn from_ggml_bytes_refuses_what_it_cannot_keep() {
    use fuel_ir::quantized::GgmlDType;
    let q8 = vec![0_u8; 34 * (256 / 32) * 4]; // 4 rows of 256 in Q8_0
    assert!(WeightStorage::from_ggml_bytes(GgmlDType::Q8_0, &q8, 256, 4).is_some());
    // In_features not a whole number of 256-blocks for a K-quant.
    assert!(WeightStorage::from_ggml_bytes(GgmlDType::Q4K, &[0; 144 * 3], 96, 8).is_none());
    // Byte count that does not match out * in / block * bytes_per_block.
    assert!(WeightStorage::from_ggml_bytes(GgmlDType::Q8_0, &q8[..q8.len() - 1], 256, 4).is_none());
    // The formats admitted with the float route (board 307) stay quantized: block bytes in, the same
    // bytes resident.
    for (dt, bytes_per_block, block) in [
        (GgmlDType::Q3K, 110, 256),
        (GgmlDType::Q5K, 176, 256),
        (GgmlDType::Iq4Nl, 18, 32),
        (GgmlDType::Iq4Xs, 136, 256),
        (GgmlDType::Iq3S, 110, 256),
    ] {
        let bytes = vec![0_u8; bytes_per_block * (512 / block) * 3];
        let w = WeightStorage::from_ggml_bytes(dt, &bytes, 512, 3);
        // Q5_K and Q3_K have a Vulkan arm since row 119 (`qmatmul_wired`), so it is kept in every build.
        if fuel_tensor::lazy::declines_unwired_vulkan_quant()
            && !matches!(
                dt,
                GgmlDType::Q5K
                    | GgmlDType::Q3K
                    | GgmlDType::Iq4Nl
                    | GgmlDType::Iq4Xs
                    | GgmlDType::Iq3S
            )
        {
            // A `vulkan` build declines them by default (no Vulkan `QMatMul` arm; opt-out `FUEL_GGML_KEEP_QUANTIZED=1`): f32 path.
            assert!(w.is_none(), "{dt:?} must decline in a vulkan build");
            continue;
        }
        let w = w.unwrap_or_else(|| panic!("{dt:?} must stay quantized"));
        assert_eq!(w.resident_bytes(), bytes.len(), "{dt:?}");
    }
    // A format with no QMatMul kernel here (Q2_K, IQ2_XXS), and f32.
    assert!(WeightStorage::from_ggml_bytes(GgmlDType::Q2K, &[0; 84 * 4], 256, 4).is_none());
    assert!(WeightStorage::from_ggml_bytes(GgmlDType::Iq2Xxs, &[0; 66 * 4], 256, 4).is_none());
    assert!(WeightStorage::from_ggml_bytes(GgmlDType::F32, &[0; 4 * 4], 4, 1).is_none());
    // Resident bytes of what it kept: the block bytes, not 4 bytes per element.
    let w = WeightStorage::from_ggml_bytes(GgmlDType::Q8_0, &q8, 256, 4).unwrap();
    assert_eq!(w.resident_bytes(), q8.len());
    assert_eq!(w.elem_count(), 256 * 4);
}
