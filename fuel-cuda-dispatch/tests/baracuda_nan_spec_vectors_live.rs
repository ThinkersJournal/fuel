// SPDX-License-Identifier: MIT OR Apache-2.0
//! The NaN SPEC VECTORS (ruling 2026-11-25, torch convention) through EVERY CUDA binding Fuel has
//! for the arg and value reduces, f32/f64/f16/bf16:
//!
//! * `ArgMaxDim` / `ArgMinDim`: a NaN WINS and the FIRST NaN's index is returned; with no NaN ties
//!   take the first index and `-0.0 == +0.0`;
//! * `MaxReduce` / `MinReduce`: any NaN in the row gives NaN; no NaN: the extreme (`+-inf` are
//!   ordinary values; `-0.0`/`+0.0` keep the FIRST-seen zero, the CPU rule).
//!
//! The same vectors are the CPU test mods `arg_nan_*` / `red_nan_*`
//! (`fuel-cpu-backend/src/byte_kernels.rs`), the Vulkan test `vulkan_arg_nan_vectors_live` and the
//! Metal test `argmax_argmin_nan_spec_vectors`. Every ALTERNATIVE registered for a key is run (the
//! PTX `fast_*` sibling AND the baracuda one), labelled by its index in the alternatives list.
//!
//! Gate: `require_gpu_run_lock`; a missing device is a FAILURE. Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch --features cuda --test baracuda_nan_spec_vectors_live -- --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::{
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use half::{bf16, f16};

const NAN: f64 = f64::NAN;
const DTYPES: [DType; 4] = [DType::F32, DType::F64, DType::F16, DType::BF16];

fn enc(dt: DType, v: &[f64]) -> Vec<u8> {
    match dt {
        DType::F32 => v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect(),
        DType::F64 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => v
            .iter()
            .flat_map(|x| f16::from_f64(*x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f64(*x).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn dec(dt: DType, b: &[u8]) -> f64 {
    match dt {
        DType::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
        DType::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()),
        DType::F16 => f16::from_le_bytes(b[..2].try_into().unwrap()).to_f64(),
        DType::BF16 => bf16::from_le_bytes(b[..2].try_into().unwrap()).to_f64(),
        _ => unreachable!(),
    }
}

fn table() -> KernelBindingTable {
    let mut t = KernelBindingTable::new();
    register_cuda_kernels(&mut t);
    register_baracuda_cuda_kernels(&mut t);
    t
}

/// Run alternative `alt` of `(op, key)` over one `[n]` row reduced on dim 0; returns the output bytes.
fn run_row(
    dev: &CudaDevice,
    t: &KernelBindingTable,
    op: OpKind,
    key: &[DType],
    alt: usize,
    in_dt: DType,
    row: &[f64],
    out_dt: DType,
) -> Result<Vec<u8>, String> {
    let alts = t.lookup_alternatives(op, key, BackendId::Cuda);
    let kernel = alts.get(alt).ok_or("no such alternative")?.kernel;
    let h =
        CudaStorageBytes::from_cpu_bytes(dev, &enc(in_dt, row)).map_err(|e| format!("{e:?}"))?;
    let src = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(h), in_dt)));
    let out = CudaStorageBytes::alloc(dev, out_dt.size_in_bytes()).map_err(|e| format!("{e:?}"))?;
    let dst = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(out), out_dt)));
    kernel(
        &[src],
        &mut [dst.clone()],
        &[Layout::contiguous(Shape::from_dims(&[row.len()]))],
        &OpParams::Reduce {
            dims: vec![0],
            keepdim: false,
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    let g = dst.read().unwrap();
    match &g.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().map_err(|e| format!("{e:?}")),
        _ => Err("not cuda".into()),
    }
}

/// `(name, row, want_argmax, want_argmin)`.
fn arg_vectors() -> Vec<(String, Vec<f64>, u32, u32)> {
    let mut long: Vec<f64> = (0..1024).map(|i| (i % 7) as f64).collect();
    long[700] = NAN;
    let long_ties: Vec<f64> = (0..1024).map(|i| (i % 7) as f64).collect();
    vec![
        ("nan-mid".into(), vec![1.0, NAN, 0.0, 5.0], 1, 1),
        ("nan-last".into(), vec![1.0, 5.0, NAN], 2, 2),
        ("nan-first".into(), vec![NAN, 1.0, 5.0], 0, 0),
        ("two-nans".into(), vec![3.0, NAN, NAN, 1.0], 1, 1),
        ("all-nan".into(), vec![NAN, NAN, NAN], 0, 0),
        ("ties".into(), vec![5.0, 5.0, 2.0, 2.0], 0, 2),
        ("zeros-a".into(), vec![-0.0, 0.0, 0.0], 0, 0),
        ("zeros-b".into(), vec![0.0, -0.0, -0.0], 0, 0),
        ("long-nan-700".into(), long, 700, 700),
        ("long-ties".into(), long_ties, 6, 0),
    ]
}

/// `(name, row, want_max, want_min)`; NaN means "must be NaN".
fn red_vectors() -> Vec<(String, Vec<f64>, f64, f64)> {
    let inf = f64::INFINITY;
    let mut long: Vec<f64> = (0..1024).map(|i| (i % 7) as f64).collect();
    long[700] = NAN;
    vec![
        ("nan-mid".into(), vec![1.0, NAN, 0.0, 5.0], NAN, NAN),
        ("nan-last".into(), vec![1.0, 5.0, NAN], NAN, NAN),
        ("nan-first".into(), vec![NAN, 1.0, 5.0], NAN, NAN),
        ("all-nan".into(), vec![NAN, NAN, NAN], NAN, NAN),
        ("long-nan-700".into(), long, NAN, NAN),
        ("plain".into(), vec![1.0, 5.0, 2.0], 5.0, 1.0),
        ("infs".into(), vec![-inf, 2.0, inf], inf, -inf),
        ("neg-infs".into(), vec![-inf, -inf], -inf, -inf),
        ("zeros-a".into(), vec![-0.0, 0.0], -0.0, -0.0),
        ("zeros-b".into(), vec![0.0, -0.0], 0.0, 0.0),
    ]
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a == b && a.is_sign_negative() == b.is_sign_negative())
}

#[test]
fn cuda_arg_and_reduce_nan_spec_vectors() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let t = table();
    let mut diverging: Vec<String> = Vec::new();
    let mut ran = 0usize;
    for dt in DTYPES {
        for (op, opname) in [(OpKind::ArgMaxDim, "argmax"), (OpKind::ArgMinDim, "argmin")] {
            let key = [dt, DType::U32];
            let n_alts = t.lookup_alternatives(op, &key, BackendId::Cuda).len();
            println!("CUDANAN {opname}/{dt:?}: {n_alts} alternative(s)");
            for alt in 0..n_alts {
                for (vname, row, want_max, want_min) in arg_vectors() {
                    let want = if opname == "argmax" {
                        want_max
                    } else {
                        want_min
                    };
                    let got = match run_row(&dev, &t, op, &key, alt, dt, &row, DType::U32) {
                        Ok(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
                        Err(e) => {
                            println!("CUDANAN {opname}/{dt:?}/alt{alt}/{vname}: ERROR {e}");
                            diverging.push(format!("{opname}/{dt:?}/alt{alt}/{vname}: error {e}"));
                            continue;
                        }
                    };
                    ran += 1;
                    let ok = got == want;
                    println!(
                        "CUDANAN {opname}/{dt:?}/alt{alt}/{vname}: got {got} want {want} {}",
                        if ok { "OK" } else { "DIVERGES" }
                    );
                    if !ok {
                        diverging.push(format!(
                            "{opname}/{dt:?}/alt{alt}/{vname}: got {got} want {want}"
                        ));
                    }
                }
            }
        }
        for (op, opname) in [(OpKind::MaxReduce, "max"), (OpKind::MinReduce, "min")] {
            let key = [dt, dt];
            let n_alts = t.lookup_alternatives(op, &key, BackendId::Cuda).len();
            println!("CUDANAN reduce-{opname}/{dt:?}: {n_alts} alternative(s)");
            for alt in 0..n_alts {
                for (vname, row, want_max, want_min) in red_vectors() {
                    let want = if opname == "max" { want_max } else { want_min };
                    let got = match run_row(&dev, &t, op, &key, alt, dt, &row, dt) {
                        Ok(b) => dec(dt, &b),
                        Err(e) => {
                            println!("CUDANAN reduce-{opname}/{dt:?}/alt{alt}/{vname}: ERROR {e}");
                            diverging.push(format!(
                                "reduce-{opname}/{dt:?}/alt{alt}/{vname}: error {e}"
                            ));
                            continue;
                        }
                    };
                    ran += 1;
                    let ok = same(got, want);
                    println!(
                        "CUDANAN reduce-{opname}/{dt:?}/alt{alt}/{vname}: got {got} want {want} {}",
                        if ok { "OK" } else { "DIVERGES" }
                    );
                    if !ok {
                        diverging.push(format!(
                            "reduce-{opname}/{dt:?}/alt{alt}/{vname}: got {got} want {want}"
                        ));
                    }
                }
            }
        }
    }
    println!("CUDANAN ran {ran} cases, {} diverge/error", diverging.len());
    assert!(ran > 0, "no CUDA binding ran");
    assert!(
        diverging.is_empty(),
        "{} CUDA case(s) diverge from the NaN spec: {diverging:?}",
        diverging.len()
    );
}
