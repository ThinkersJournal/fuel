// SPDX-License-Identifier: MIT OR Apache-2.0
//! A GGUF whose projections are Q3_K, Q5_K, IQ4_NL, IQ4_XS or IQ3_S must realize on Vulkan.
//!
//! `WeightStorage::from_ggml_bytes` keeps those types quantized (#686), the Vulkan `QMatMul` has no arm
//! for them, and the planner still places the op on Vulkan, so before the loader declined them in
//! `vulkan` builds the realize returned `QuantType Q5K not wired on Vulkan` (measured at `7ecc79f6e`,
//! `examples/gguf_unwired_quant_probe.rs`). This test is that probe's model path as a test:
//!
//! - the five types: the one-layer llama of `tests/common/gguf_synth.rs` realizes on the Vulkan device and
//!   the projections are resident as f32 (the decline), `QMatMul` launches are 0;
//! - the controls Q4_K and Q6_K: the projections stay resident as blocks and all 8 `QMatMul` launches
//!   of the model run on Vulkan (the decline must not reach a wired type).
//!
//! Gate: the machine-wide GPU lock (`require_gpu_run_lock`); a missing Vulkan device is a FAILURE. Not
//! `#[ignore]`d. Run on the GPU runner with
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-transformers --features dev-tools,vulkan \
//!  --test gguf_unwired_quant_vulkan_live -- --nocapture --test-threads=1`.
#![cfg(feature = "vulkan")]

#[path = "common/gguf_synth.rs"]
mod synth;

use fuel_dispatch::compiled::op_profile;
use fuel_quantized::iq4::{BlockIq4Nl, BlockIq4Xs};
use fuel_quantized::{BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, GgmlType};
use synth::{files_with, load, projection_weights, projections, quantize};

type Quant = Box<dyn Fn(&[f32]) -> Vec<u8>>;

fn q<B: GgmlType>() -> Quant {
    Box::new(|w| quantize::<B>(w).0)
}

/// IQ3_S is decode-only (Fuel cannot quantize to it): 110-byte blocks, a finite f16 scale (0.05 =
/// 0x2A66) then pseudo-random bytes.
fn iq3s() -> Quant {
    Box::new(|w| {
        let mut s = 0x1234_5678_u32;
        let mut out = Vec::new();
        for _ in 0..w.len() / 256 {
            out.extend_from_slice(&[0x66, 0x2A]);
            for _ in 0..108 {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                out.push((s >> 16) as u8);
            }
        }
        out
    })
}

/// `(name, GGML type code, quantizer, kept quantized on Vulkan)`.
fn types() -> Vec<(&'static str, u32, Quant, bool)> {
    vec![
        ("Q4_K", 12, q::<BlockQ4K>(), true),
        ("Q6_K", 14, q::<BlockQ6K>(), true),
        ("Q3_K", 11, q::<BlockQ3K>(), false),
        ("Q5_K", 13, q::<BlockQ5K>(), false),
        ("IQ4_NL", 20, q::<BlockIq4Nl>(), false),
        ("IQ4_XS", 23, q::<BlockIq4Xs>(), false),
        ("IQ3_S", 21, iq3s(), false),
    ]
}

#[test]
fn unwired_gguf_types_realize_on_vulkan_and_wired_ones_stay_quantized() {
    fuel_test_support::require_gpu_run_lock();
    let dev = fuel_tensor::vulkan_backend::new_device_from_env()
        .expect("a Vulkan device (a missing adapter is a failure, GAP-157)");
    let f32_bytes: usize = projections().iter().map(|(_, o, i)| o * i * 4).sum();
    for (name, code, quant, wired) in types() {
        let (file, qbytes) = files_with(code, &*quant);
        let model = load(&file, "unwired");
        let fwd = model.forward(&[1_u32, 5, 9], 0).expect("forward");
        let g = fwd.graph_tensor();
        let graph = g.graph().clone();
        op_profile::set_enabled(true);
        let _ = op_profile::take();
        let out = fuel_tensor::pipelined_bridge::realize_one_as::<f32>(&graph, g.id(), &dev);
        let rows = op_profile::take();
        op_profile::set_enabled(false);
        let out = out.unwrap_or_else(|e| panic!("{name}: realize on Vulkan failed: {e}"));
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{name}: non-finite logit"
        );
        assert!(out.iter().any(|v| *v != 0.0), "{name}: all-zero logits");

        let resident: usize = projection_weights(&model)
            .iter()
            .map(|w| w.resident_bytes())
            .sum();
        if wired {
            assert_eq!(resident, qbytes, "{name}: a wired type must stay as blocks");
        } else {
            assert_eq!(
                resident, f32_bytes,
                "{name}: an unwired type must load as f32 in a vulkan build"
            );
        }

        let count = |backend: &str, only_q: bool| -> u64 {
            rows.iter()
                .filter(|r| r.1.contains(backend) && (!only_q || r.0.contains("QMatMul")))
                .map(|r| r.2)
                .sum()
        };
        let (vk_q, cpu_q) = (count("Vulkan", true), count("Cpu", true));
        // Control: ops did run on Vulkan, so "0 QMatMul" is not "nothing ran".
        assert!(count("Vulkan", false) > 0, "{name}: nothing ran on Vulkan");
        assert_eq!(cpu_q, 0, "{name}: QMatMul fell to the CPU arm");
        if wired {
            assert_eq!(vk_q, 8, "{name}: all 8 QMatMul launches must be on Vulkan");
        } else {
            assert_eq!(vk_q, 0, "{name}: an f32 weight must not use QMatMul");
        }
        println!("{name}: resident {resident} B, QMatMul vulkan={vk_q} cpu={cpu_q}: ok");
    }
}
