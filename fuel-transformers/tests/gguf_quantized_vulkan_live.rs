// SPDX-License-Identifier: MIT OR Apache-2.0
//! A GGUF whose projections are Q4_K, Q6_K, Q3_K, Q5_K, IQ4_NL, IQ4_XS or IQ3_S must realize on Vulkan with
//! the projections resident as blocks and every `QMatMul` launched on Vulkan.
//!
//! History: `WeightStorage::from_ggml_bytes` kept Q3_K, Q5_K, IQ4_NL, IQ4_XS and IQ3_S quantized from #686
//! on while the Vulkan `QMatMul` had no arm for them, so the realize returned `QuantType Q5K not wired on
//! Vulkan` (measured at `7ecc79f6e`, `examples/gguf_unwired_quant_probe.rs`); hotfix #706 declined them in
//! `vulkan` builds, and ROADMAP row 119 wired all five (fused GLSL gemv kernels) and removed the decline.
//! This test is the model-path regression for that class: the one-layer llama of `tests/common/gguf_synth.rs`
//! per type, projections resident as blocks (`resident_bytes` equals the quantized file bytes) and all 8
//! `QMatMul` launches of the model on Vulkan, none on the CPU arm.
//!
//! Gate: the machine-wide GPU lock (`require_gpu_run_lock`); a missing Vulkan device is a FAILURE. Not
//! `#[ignore]`d. Run on the GPU runner with
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-transformers --features dev-tools,vulkan //!  --test gguf_quantized_vulkan_live -- --nocapture --test-threads=1`.
#![cfg(feature = "vulkan")]

#[path = "common/gguf_synth.rs"]
mod synth;

use fuel_dispatch::compiled::op_profile;
use fuel_quantized::iq4::{BlockIq4Nl, BlockIq4Xs};
use fuel_quantized::{BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, GgmlType};
use synth::{
    H, LAYERS, Lcg, T, VOCAB, build_gguf, f32_tensor, files, load, projection_weights, projections,
    quantize,
};

/// The quantized-only file of `synth::files`, with the caller supplying the quantizer (`quant` maps
/// `[out, in]` f32 values to block bytes), for a type Fuel cannot quantize to (IQ3_S is decode-only).
/// Same tensors and seed as `synth::files`. Returns `(file, quantized bytes of the projections)`.
fn files_with(code: u32, quant: &dyn Fn(&[f32]) -> Vec<u8>) -> (Vec<u8>, usize) {
    let mut rng = Lcg(0x5eed);
    let mut t = vec![f32_tensor(
        "token_embd.weight",
        &[H as u64, VOCAB as u64],
        &rng.vec(H * VOCAB),
    )];
    for l in 0..LAYERS {
        for n in ["attn_norm", "ffn_norm"] {
            t.push(f32_tensor(
                &format!("blk.{l}.{n}.weight"),
                &[H as u64],
                &vec![1.0; H],
            ));
        }
    }
    t.push(f32_tensor("output_norm.weight", &[H as u64], &vec![1.0; H]));
    let mut qbytes = 0;
    for (name, o, i) in projections() {
        let data = quant(&rng.vec(o * i));
        qbytes += data.len();
        t.push(T {
            name,
            dims: vec![i as u64, o as u64],
            code,
            data,
        });
    }
    (build_gguf(&t), qbytes)
}

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

/// `(name, GGML type code, quantizer)`.
fn types() -> Vec<(&'static str, u32, Quant)> {
    vec![
        ("Q4_K", 12, q::<BlockQ4K>()),
        ("Q6_K", 14, q::<BlockQ6K>()),
        ("Q3_K", 11, q::<BlockQ3K>()),
        ("Q5_K", 13, q::<BlockQ5K>()),
        ("IQ4_NL", 20, q::<BlockIq4Nl>()),
        ("IQ4_XS", 23, q::<BlockIq4Xs>()),
        ("IQ3_S", 21, iq3s()),
    ]
}

#[test]
fn every_kept_gguf_type_realizes_on_vulkan_as_blocks() {
    fuel_test_support::require_gpu_run_lock();
    let dev = fuel_tensor::vulkan_backend::new_device_from_env()
        .expect("a Vulkan device (a missing adapter is a failure, GAP-157)");
    for (name, code, quant) in types() {
        let (file, qbytes) = if code == 12 || code == 14 {
            // The controls build their file with the shared helper the other tests use.
            let (f, _plain, q) = if code == 12 {
                files::<BlockQ4K>(code, 1.0)
            } else {
                files::<BlockQ6K>(code, 1.0)
            };
            (f, q)
        } else {
            files_with(code, &*quant)
        };
        let model = load(&file, "quantized");
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
        assert_eq!(resident, qbytes, "{name}: a kept type must stay as blocks");

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
        assert_eq!(vk_q, 8, "{name}: all 8 QMatMul launches must be on Vulkan");
        println!("{name}: resident {resident} B, QMatMul vulkan={vk_q} cpu={cpu_q}: ok");
    }
}
