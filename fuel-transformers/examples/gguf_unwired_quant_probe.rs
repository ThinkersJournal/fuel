// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 124 follow-up: what does a GGUF weight type that `WeightStorage::from_ggml_bytes` keeps
//! quantized but `qmatmul_vk` has no Vulkan arm for do on a Vulkan device? For each type this prints
//! whether the realize (a) succeeds with every `QMatMul` on Vulkan, (b) succeeds with some or all
//! `QMatMul` on the CPU arm, or (c) returns an error or panics (the message is printed).
//!
//! Controls: Q4_K and Q6_K (wired on Vulkan, must come out (a)). Subjects: Q3_K, Q5_K, IQ4_NL, IQ4_XS and
//! IQ3_S (IQ3_S is decode-only in Fuel: its bytes are synthetic with a finite f16 scale, so its
//! numbers say nothing about quantization quality). Two shapes per type: a bare one-op graph at
//! `n = 256` and `n = 255` (odd), and the one-layer llama of `tests/common/gguf_synth.rs`.
//!
//! ```text
//! /c/gpu-run/bin/gpu-run.exe -Project fuel -- <built exe>
//! ```
//!
//! Build: `cargo build -p fuel-transformers --features dev-tools,vulkan --example gguf_unwired_quant_probe`.
//! Exit code is nonzero if a control does not come out (a), so a probe that is blind cannot pass.

#[cfg(feature = "vulkan")]
#[path = "../tests/common/gguf_synth.rs"]
mod synth;

#[cfg(feature = "vulkan")]
mod probe {
    use super::synth::{files_with, load, projection_weights, quantize};
    use fuel_dispatch::compiled::op_profile;
    use fuel_graph::QuantType;
    use fuel_quantized::iq4::{BlockIq4Nl, BlockIq4Xs};
    use fuel_quantized::{BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, GgmlType};
    use fuel_tensor::Device;
    use fuel_tensor::lazy::Tensor;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    type Quant = Box<dyn Fn(&[f32]) -> Vec<u8>>;

    struct Ty {
        name: &'static str,
        code: u32,
        qtype: QuantType,
        control: bool,
        quant: Quant,
    }

    fn q<B: GgmlType>() -> Quant {
        Box::new(|w| quantize::<B>(w).0)
    }

    /// IQ3_S blocks are 110 bytes: an f16 scale `d` then packed indices and signs. Fuel cannot quantize
    /// to it, so the bytes are pseudo-random after a fixed finite scale (0.05 as f16 = 0x2A66).
    fn iq3s() -> Quant {
        Box::new(|w| {
            let blocks = w.len() / 256;
            let mut s = 0x1234_5678_u32;
            let mut out = Vec::with_capacity(blocks * 110);
            for _ in 0..blocks {
                out.extend_from_slice(&[0x66, 0x2A]);
                for _ in 0..108 {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    out.push((s >> 16) as u8);
                }
            }
            out
        })
    }

    fn types() -> Vec<Ty> {
        vec![
            Ty {
                name: "Q4_K (control)",
                code: 12,
                qtype: QuantType::Q4_K_M,
                control: true,
                quant: q::<BlockQ4K>(),
            },
            Ty {
                name: "Q6_K (control)",
                code: 14,
                qtype: QuantType::Q6K,
                control: true,
                quant: q::<BlockQ6K>(),
            },
            Ty {
                name: "Q3_K",
                code: 11,
                qtype: QuantType::Q3K,
                control: false,
                quant: q::<BlockQ3K>(),
            },
            Ty {
                name: "Q5_K",
                code: 13,
                qtype: QuantType::Q5K,
                control: false,
                quant: q::<BlockQ5K>(),
            },
            Ty {
                name: "IQ4_NL",
                code: 20,
                qtype: QuantType::Iq4Nl,
                control: false,
                quant: q::<BlockIq4Nl>(),
            },
            Ty {
                name: "IQ4_XS",
                code: 23,
                qtype: QuantType::Iq4Xs,
                control: false,
                quant: q::<BlockIq4Xs>(),
            },
            Ty {
                name: "IQ3_S",
                code: 21,
                qtype: QuantType::Iq3S,
                control: false,
                quant: iq3s(),
            },
        ]
    }

    /// `Ok(values)` or the error or panic text.
    fn realize(t: &Tensor, dev: &Device) -> Result<Vec<f32>, String> {
        let g = t.graph_tensor();
        let graph = g.graph().clone();
        let id = g.id();
        match catch_unwind(AssertUnwindSafe(|| {
            fuel_tensor::pipelined_bridge::realize_one_as::<f32>(&graph, id, dev)
        })) {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(format!("error: {e}")),
            Err(p) => Err(format!(
                "panic: {}",
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default()
            )),
        }
    }

    fn counts(rows: &[(String, String, u64, std::time::Duration)]) -> (u64, u64, u64) {
        let sum = |be: &str, only_q: bool| -> u64 {
            rows.iter()
                .filter(|r| r.1.contains(be) && (!only_q || r.0.contains("QMatMul")))
                .map(|r| r.2)
                .sum()
        };
        (sum("Vulkan", true), sum("Cpu", true), sum("Vulkan", false))
    }

    /// (a) = 'A', (b) = 'B', (c) = 'C'.
    fn verdict(r: &Result<Vec<f32>, String>, vk_q: u64, cpu_q: u64) -> char {
        match r {
            Err(_) => 'C',
            Ok(_) if cpu_q == 0 && vk_q > 0 => 'A',
            Ok(_) => 'B',
        }
    }

    fn isolated(ty: &Ty, (m, k, n): (usize, usize, usize), dev: &Device) -> char {
        use fuel_ir::Shape;
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.021).sin() * 0.7)
            .collect();
        let mut bytes = (ty.quant)(&w_f32);
        bytes.resize(bytes.len().div_ceil(4) * 4, 0); // whole U32 words, as from_ggml_bytes pads
        let w_u32: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
        let x = Tensor::from_f32(a, Shape::from_dims(&[m, k]), &Device::cpu()).unwrap();
        let w = x
            .const_u32_like(w_u32, Shape::from_dims(&[bytes.len().div_ceil(4)]))
            .unwrap();
        let y = match x.qmatmul(&w, ty.qtype, k, n) {
            Ok(y) => y,
            Err(e) => {
                println!("  isolated {} m={m} k={k} n={n}: BUILD error: {e}", ty.name);
                return 'C';
            }
        };
        op_profile::set_enabled(true);
        let _ = op_profile::take();
        let r = realize(&y, dev);
        let rows = op_profile::take();
        op_profile::set_enabled(false);
        let (vk_q, cpu_q, vk_all) = counts(&rows);
        let v = verdict(&r, vk_q, cpu_q);
        let detail = match &r {
            Ok(o) => format!("out[0]={:.4}", o[0]),
            Err(e) => e.clone(),
        };
        println!(
            "  isolated {} m={m} k={k} n={n}: ({v}) QMatMul vulkan={vk_q} cpu={cpu_q}, all ops vulkan={vk_all}; {detail}",
            ty.name
        );
        v
    }

    fn model(ty: &Ty, dev: &Device) -> char {
        let (file, qbytes) = files_with(ty.code, &*ty.quant);
        let loaded = catch_unwind(AssertUnwindSafe(|| load(&file, "probe")));
        let m = match loaded {
            Ok(m) => m,
            Err(_) => {
                println!("  model {}: LOAD panicked", ty.name);
                return 'C';
            }
        };
        let resident: usize = projection_weights(&m)
            .iter()
            .map(|w| w.resident_bytes())
            .sum();
        println!(
            "  model {}: projections resident {resident} B (quantized file bytes {qbytes}; equal means kept as blocks)",
            ty.name
        );
        let toks = [1_u32, 5, 9];
        let fwd = match m.forward(&toks, 0) {
            Ok(f) => f,
            Err(e) => {
                println!("  model {}: forward build error: {e}", ty.name);
                return 'C';
            }
        };
        op_profile::set_enabled(true);
        let _ = op_profile::take();
        let r = realize(&fwd, dev);
        let rows = op_profile::take();
        op_profile::set_enabled(false);
        let (vk_q, cpu_q, vk_all) = counts(&rows);
        let v = verdict(&r, vk_q, cpu_q);
        let detail = match &r {
            Ok(o) => format!("{} logits, first {:.4}", o.len(), o[0]),
            Err(e) => e.clone(),
        };
        println!(
            "  model {}: ({v}) QMatMul vulkan={vk_q} cpu={cpu_q}, all ops vulkan={vk_all}; {detail}",
            ty.name
        );
        v
    }

    pub fn main() {
        let dev = fuel_tensor::vulkan_backend::new_device_from_env().expect("a Vulkan device");
        let vk = fuel_tensor::vulkan_backend::as_device(&dev).expect("a Vulkan device");
        println!("adapter: [{}] {}", vk.gpu_id, vk.device_name);
        println!(
            "legend: (A) realize ok, every QMatMul on Vulkan; (B) ok, some QMatMul on the CPU arm; (C) error or panic"
        );
        let mut blind = false;
        for ty in types() {
            println!("{}:", ty.name);
            let mut vs = Vec::new();
            for shape in [(1, 256, 256), (1, 256, 255), (3, 256, 256)] {
                vs.push(isolated(&ty, shape, &dev));
            }
            vs.push(model(&ty, &dev));
            let s: String = vs.iter().collect();
            println!(
                "  => {} verdicts [isolated n=256 m=1, n=255 m=1, n=256 m=3, model]: {s}",
                ty.name
            );
            if ty.control && s.chars().any(|c| c != 'A') {
                println!(
                    "  FAIL: control {} did not come out (A) everywhere",
                    ty.name
                );
                blind = true;
            }
        }
        if blind {
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "vulkan")]
fn main() {
    probe::main();
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("build with --features dev-tools,vulkan");
    std::process::exit(2);
}
