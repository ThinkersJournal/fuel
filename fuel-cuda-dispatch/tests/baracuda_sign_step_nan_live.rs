// SPDX-License-Identifier: MIT OR Apache-2.0
//! `sign(NaN)` and `step(NaN)` on every CUDA binding (f32/f64/f16/bf16, each alternative): KISS 6.2-0009
//! pins them at ZERO (non-primitive ops follow their decomposition: `sign(NaN) = 0`, `step(NaN) = 0`);
//! torch differs (`torch.sign(NaN) = NaN`) and KISS wins (pm-kernels 2026-11-26). The CPU kernel and Metal
//! agree. Inputs `[NaN, 0.5, -2.0, 0.0]`: sign -> `[0, 1, -1, 0]`; step -> `[0, 1, 0, _]` (`step(0)` is not
//! asserted). A kernel that returns NaN for NaN here DIVERGES from the spec.
//!
//! Gate: `require_gpu_run_lock`. Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch
//! --features cuda --test baracuda_sign_step_nan_live -- --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::{
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use half::{bf16, f16};

const DTYPES: [DType; 4] = [DType::F32, DType::F64, DType::F16, DType::BF16];

fn enc(dt: DType, v: &[f32]) -> Vec<u8> {
    match dt {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F64 => v.iter().flat_map(|x| (*x as f64).to_le_bytes()).collect(),
        DType::F16 => v
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn dec(dt: DType, b: &[u8]) -> Vec<f32> {
    match dt {
        DType::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F64 => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        DType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        DType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        _ => unreachable!(),
    }
}

fn run(
    dev: &CudaDevice,
    t: &KernelBindingTable,
    op: OpKind,
    dt: DType,
    alt: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let alts = t.lookup_alternatives(op, &[dt, dt], BackendId::Cuda);
    let kernel = alts.get(alt).ok_or("no such alternative")?.kernel;
    let h = CudaStorageBytes::from_cpu_bytes(dev, &enc(dt, input)).map_err(|e| format!("{e:?}"))?;
    let src = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(h), dt)));
    let out = CudaStorageBytes::alloc(dev, input.len() * dt.size_in_bytes())
        .map_err(|e| format!("{e:?}"))?;
    let dst = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(out), dt)));
    kernel(&[src], &mut [dst.clone()], &[], &OpParams::None).map_err(|e| format!("{e:?}"))?;
    let g = dst.read().unwrap();
    match &g.inner {
        BackendStorage::Cuda(c) => Ok(dec(dt, &c.to_cpu_bytes().map_err(|e| format!("{e:?}"))?)),
        _ => Err("not cuda".into()),
    }
}

#[test]
fn cuda_sign_and_step_of_nan_are_zero() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let mut t = KernelBindingTable::new();
    register_cuda_kernels(&mut t);
    register_baracuda_cuda_kernels(&mut t);
    let input = [f32::NAN, 0.5, -2.0, 0.0];
    let cases: [(&str, OpKind, &[Option<f32>]); 2] = [
        (
            "sign",
            OpKind::SignElementwise,
            &[Some(0.0), Some(1.0), Some(-1.0), Some(0.0)],
        ),
        (
            "step",
            OpKind::StepElementwise,
            &[Some(0.0), Some(1.0), Some(0.0), None],
        ),
    ];
    let (mut bad, mut ran) = (Vec::new(), 0usize);
    for dt in DTYPES {
        for (name, op, want) in cases {
            let n = t.lookup_alternatives(op, &[dt, dt], BackendId::Cuda).len();
            if n == 0 {
                println!("CUDASIGN {name}/{dt:?}: UNBOUND (no CUDA binding)");
                continue;
            }
            for alt in 0..n {
                match run(&dev, &t, op, dt, alt, &input) {
                    Err(e) => println!(
                        "CUDASIGN {name}/{dt:?}/alt{alt}: UNREACHABLE ({})",
                        e.lines().next().unwrap_or("")
                    ),
                    Ok(got) => {
                        ran += 1;
                        let ok = got.iter().zip(want).all(|(g, w)| match w {
                            None => true,
                            Some(x) => !g.is_nan() && g == x,
                        });
                        println!(
                            "CUDASIGN {name}/{dt:?}/alt{alt}: got {got:?} {}",
                            if ok { "OK" } else { "DIVERGES" }
                        );
                        if !ok {
                            bad.push(format!("{name}/{dt:?}/alt{alt}: {got:?}"));
                        }
                    }
                }
            }
        }
    }
    println!("CUDASIGN ran {ran} cases, {} diverge", bad.len());
    assert!(ran > 0, "no CUDA sign/step binding ran");
    assert!(bad.is_empty(), "{bad:?}");
}

/// `gelu_tanh` on every dtype: 1.0 -> 0.841192 (tanh approximation), NaN -> NaN, 0 -> 0. Turns the
/// GAP-411 "derived" claim for the half and f64 sections into a measurement.
#[test]
fn cuda_gelu_tanh_vectors_hold_on_every_dtype() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let mut t = KernelBindingTable::new();
    register_cuda_kernels(&mut t);
    register_baracuda_cuda_kernels(&mut t);
    let input = [1.0f32, f32::NAN, 0.0];
    let (mut bad, mut ran) = (Vec::new(), 0usize);
    for dt in DTYPES {
        let tol = match dt {
            DType::F16 => 2e-3,
            DType::BF16 => 1e-2,
            _ => 1e-5,
        };
        let op = OpKind::GeluTanhElementwise;
        let n = t.lookup_alternatives(op, &[dt, dt], BackendId::Cuda).len();
        assert!(n > 0, "gelu_tanh/{dt:?}: no CUDA binding");
        for alt in 0..n {
            let got = run(&dev, &t, op, dt, alt, &input).expect("gelu_tanh runs");
            ran += 1;
            let ok = (got[0] - 0.841192).abs() <= tol && got[1].is_nan() && got[2] == 0.0;
            println!(
                "CUDAGELU gelu_tanh/{dt:?}/alt{alt}: got {got:?} {}",
                if ok { "OK" } else { "DIVERGES" }
            );
            if !ok {
                bad.push(format!("{dt:?}/alt{alt}: {got:?}"));
            }
        }
    }
    println!("CUDAGELU ran {ran} cases, {} diverge", bad.len());
    assert!(bad.is_empty(), "{bad:?}");
}
