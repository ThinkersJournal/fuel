// SPDX-License-Identifier: MIT OR Apache-2.0
//! Every GGUF dtype `WeightStorage::from_ggml_bytes` can keep resident has a Vulkan `QMatMul` arm
//! (ROADMAP row 119; GAP-397).
//!
//! The #686 class: the loader kept a type quantized while `qmatmul_vk` had no arm for it, so a model of
//! that type failed at realize on Vulkan ("not wired on Vulkan"). The decline that guarded it is gone;
//! this test is what keeps the class closed. `all()` lists EVERY `GgmlDType` through a `match` with no
//! wildcard, so a new dtype is a compile error here until it is listed; for each one, the loader is asked
//! with a well-formed block stream, and if it keeps the weight, `qmatmul_wired` must list its `QuantType`.
//! The population is printed ("N types, M kept (all wired), K declined").
#![cfg(feature = "vulkan")]

use fuel_ir::quantized::GgmlDType;
use fuel_tensor::lazy::WeightStorage;
use fuel_vulkan_dispatch::vulkan_dispatch::qmatmul::qmatmul_wired;

/// Every dtype, named. The `match` is exhaustive: a new `GgmlDType` variant does not compile until it is
/// added here (and in `from_ggml_bytes`, which has no wildcard arm either).
fn all() -> Vec<GgmlDType> {
    let v = vec![
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
        GgmlDType::Iq4Nl,
        GgmlDType::Iq4Xs,
        GgmlDType::Iq2Xxs,
        GgmlDType::Iq2Xs,
        GgmlDType::Iq3Xxs,
        GgmlDType::Iq1S,
        GgmlDType::Iq3S,
        GgmlDType::Iq2S,
        GgmlDType::Iq1M,
        GgmlDType::Tq1_0,
        GgmlDType::Tq2_0,
        GgmlDType::Mxfp4,
        GgmlDType::Nvfp4,
    ];
    for d in &v {
        match d {
            GgmlDType::F32
            | GgmlDType::F16
            | GgmlDType::BF16
            | GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::Q8K
            | GgmlDType::Iq4Nl
            | GgmlDType::Iq4Xs
            | GgmlDType::Iq2Xxs
            | GgmlDType::Iq2Xs
            | GgmlDType::Iq3Xxs
            | GgmlDType::Iq1S
            | GgmlDType::Iq3S
            | GgmlDType::Iq2S
            | GgmlDType::Iq1M
            | GgmlDType::Tq1_0
            | GgmlDType::Tq2_0
            | GgmlDType::Mxfp4
            | GgmlDType::Nvfp4 => {}
        }
    }
    v
}

/// `(kept dtypes whose QuantType `wired` rejects, number kept, number declined)`.
fn survey(wired: impl Fn(fuel_graph::QuantType) -> bool) -> (Vec<GgmlDType>, usize, usize) {
    let (mut unwired, mut kept, mut declined) = (Vec::new(), 0, 0);
    for dt in all() {
        // Two output rows of 512 inputs: a whole number of blocks for every block size (1, 32, 64, 256).
        let (rows, cols) = (2usize, 512usize);
        let bytes = vec![0_u8; rows * (cols / dt.block_size()) * dt.type_size()];
        match WeightStorage::from_ggml_bytes(dt, &bytes, cols, rows) {
            Some(WeightStorage::Ggml { qtype, .. }) => {
                kept += 1;
                if !wired(qtype) {
                    unwired.push(dt);
                }
            }
            Some(other) => panic!("{dt:?}: unexpected storage {:?}", other.resident_bytes()),
            None => declined += 1,
        }
    }
    (unwired, kept, declined)
}

#[test]
fn every_dtype_the_loader_keeps_has_a_vulkan_arm() {
    let n = all().len();
    let (unwired, kept, declined) = survey(qmatmul_wired);
    println!("{n} GgmlDType variants: {kept} kept (all wired), {declined} declined by name");
    assert_eq!(kept + declined, n);
    assert!(
        unwired.is_empty(),
        "kept but not wired on Vulkan: {unwired:?}"
    );
    // Controls: the survey sees the kept set (8: Q8_0, Q4_K, Q6_K, Q3_K, Q5_K, IQ4_NL, IQ4_XS, IQ3_S) and
    // the declines (Q4_0 has its own variant; the rest have no kernel).
    assert_eq!(
        kept, 8,
        "kept population changed: update this test and the loader doc together"
    );
    assert_eq!(declined, n - 8);
}

/// The sabotage sibling: with one kept type missing from the wired predicate (Q3_K), the survey names it.
#[test]
fn a_kept_type_without_an_arm_is_caught() {
    let (unwired, _, _) = survey(|q| qmatmul_wired(q) && q != fuel_graph::QuantType::Q3K);
    assert_eq!(unwired, vec![GgmlDType::Q3K]);
}
