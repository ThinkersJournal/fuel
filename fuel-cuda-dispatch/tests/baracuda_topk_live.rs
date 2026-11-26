// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-CUDA test for the `OpKind::TopKLastDim` binding (ROADMAP row 172 item 2
//! step A): baracuda's block-bitonic `topk` reached through the contract-registered
//! wrapper, against the CPU kernel (`fuel_cpu_backend::byte_kernels::top_k_last_dim_*`),
//! which pins NaN-largest and ties-lower-index-first.
//!
//! Not `#[ignore]`d: the file is `cfg(feature = "cuda")`, so no CI leg builds it,
//! and `require_gpu_run_lock` plus `CudaDevice::new` fail loudly without the
//! wrapper or a device.
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch
//! --features cuda --test baracuda_topk_live`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cpu_backend::CpuStorageBytes;
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::{
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};

/// `(values, indices)` decoded from a bundled `[values ; pad to 4 ; indices]` buffer.
fn decode_f32(bytes: &[u8], rows: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    let vb = rows * k * 4;
    let v = bytes[..vb]
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let i = bytes[vb..vb + rows * k * 4]
        .chunks(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (v, i)
}

fn run_cuda_f32(x: &[f32], rows: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    let kernel = table
        .lookup(
            OpKind::TopKLastDim,
            &[DType::F32, DType::F32],
            BackendId::Cuda,
        )
        .expect("TopKLastDim F32 binding must be registered on CUDA");
    let input = {
        let cuda = CudaStorageBytes::from_cpu_bytes(&dev, bytemuck::cast_slice(x)).expect("h2d");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(cuda),
            DType::F32,
        )))
    };
    let out = {
        let (total, _) = fuel_cuda_backend::baracuda::topk::bundle_layout(4, rows, k);
        let buf = CudaStorageBytes::alloc(&dev, total).expect("alloc");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(buf),
            DType::F32,
        )))
    };
    kernel(
        &[input],
        &mut [out.clone()],
        &[],
        &OpParams::TopKLastDim { rows, n, k },
    )
    .expect("top_k_last_dim");
    let guard = out.read().unwrap();
    let BackendStorage::Cuda(c) = &guard.inner else {
        panic!("not on CUDA")
    };
    decode_f32(&c.to_cpu_bytes().expect("d2h"), rows, k)
}

fn run_cpu_f32(x: &[f32], rows: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
    let (total, _) = fuel_cuda_backend::baracuda::topk::bundle_layout(4, rows, k);
    let mut out = CpuStorageBytes::from_zero_bytes(total);
    fuel_cpu_backend::byte_kernels::top_k_last_dim_f32(
        &CpuStorageBytes::from_slice(x),
        &mut out,
        rows,
        n,
        k,
    )
    .expect("cpu kernel");
    decode_f32(out.bytes(), rows, k)
}

/// Distinct values: the CUDA kernel and the CPU kernel agree exactly, on a shape
/// with several rows, and the result is descending (a sabotage of the order would
/// move the indices).
#[test]
fn cuda_topk_f32_matches_the_cpu_kernel_on_distinct_values() {
    let (rows, n, k) = (5usize, 100usize, 7usize);
    let x: Vec<f32> = (0..rows * n)
        .map(|i| (((i * 2654435761usize) >> 7) % 100_003) as f32 * 0.001 - 30.0)
        .collect();
    let mut sorted = x.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        x.len(),
        "the fixture must have distinct values"
    );
    let got = run_cuda_f32(&x, rows, n, k);
    let want = run_cpu_f32(&x, rows, n, k);
    assert_eq!(got, want);
}

/// Over-cap shapes are a typed error before any launch (row_len > 1024, k > 64):
/// there is no decline to the CPU path in the binding table.
#[test]
fn cuda_topk_over_cap_shapes_are_a_typed_error() {
    fuel_test_support::require_gpu_run_lock();
    assert!(fuel_cuda_backend::baracuda::topk::check_caps(1, 2000, 3).is_err());
    assert!(fuel_cuda_backend::baracuda::topk::check_caps(1, 200, 65).is_err());
}

/// Ties and NaN at baracuda `0.1.0-alpha.188` (the top-k NaN/ties fix, baracuda #429).
/// At alpha.186 these two vectors DIVERGED from the CPU kernel (GAP-402); this test now
/// requires AGREEMENT with the CPU kernel (NaN largest, ties lower index first).
#[test]
fn cuda_topk_f32_pinned_vectors_match_the_cpu_kernel() {
    let x = [3.0_f32, 5.0, 5.0, 1.0, 5.0, 2.0, 4.0, 5.0];
    let got = run_cuda_f32(&x, 1, 8, 3);
    let want = run_cpu_f32(&x, 1, 8, 3);
    assert_eq!(
        want.1,
        vec![1, 2, 4],
        "the CPU kernel takes the lowest tied indices"
    );
    assert_eq!(got, want, "ties");
    let nanx = [1.0_f32, f32::NAN, 9.0, 2.0];
    let got = run_cuda_f32(&nanx, 1, 4, 2);
    let want = run_cpu_f32(&nanx, 1, 4, 2);
    assert!(want.0[0].is_nan() && want.0[1] == 9.0 && want.1 == vec![1, 2]);
    assert!(got.0[0].is_nan() && got.0[1] == 9.0, "NaN row: {got:?}");
    assert_eq!(got.1, want.1, "NaN row indices");
}

/// Property test: random rows over a SMALL alphabet (so ties are everywhere) with NaN
/// injected at 1 in 8, across row lengths up to the 1024 cap and k up to the 64 cap,
/// CUDA against the CPU kernel, values by `is_nan`-or-equal and indices exactly.
/// Rows tested are counted and printed (the population is stated, not implied).
#[test]
fn cuda_topk_f32_random_nan_and_tie_rows_match_the_cpu_kernel() {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut rows_total = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for &(n, k) in &[
        (8usize, 3usize),
        (33, 1),
        (64, 8),
        (100, 7),
        (257, 32),
        (512, 64),
        (1000, 17),
        (1024, 64),
        (1024, 1),
        (65, 64),
    ] {
        for case in 0..12 {
            let rows = 32usize;
            let x: Vec<f32> = (0..rows * n)
                .map(|_| {
                    let r = next();
                    if r % 8 == 0 {
                        f32::NAN
                    } else {
                        ((r >> 8) % 6) as f32 - 2.0
                    }
                })
                .collect();
            let got = run_cuda_f32(&x, rows, n, k);
            let want = run_cpu_f32(&x, rows, n, k);
            rows_total += rows;
            let vals_ok = got
                .0
                .iter()
                .zip(&want.0)
                .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b);
            if !vals_ok || got.1 != want.1 {
                let first = (0..rows)
                    .find(|r| {
                        got.1[r * k..(r + 1) * k] != want.1[r * k..(r + 1) * k]
                            || got.0[r * k..(r + 1) * k]
                                .iter()
                                .zip(&want.0[r * k..(r + 1) * k])
                                .any(|(a, b)| !((a.is_nan() && b.is_nan()) || a == b))
                    })
                    .unwrap();
                bad.push(format!(
                    "n={n} k={k} case={case} row={first}: cuda {:?}/{:?} cpu {:?}/{:?} input {:?}",
                    &got.0[first * k..(first + 1) * k],
                    &got.1[first * k..(first + 1) * k],
                    &want.0[first * k..(first + 1) * k],
                    &want.1[first * k..(first + 1) * k],
                    &x[first * n..(first * n + n).min(first * n + 40)]
                ));
            }
        }
    }
    println!(
        "topk property: {rows_total} rows, {} failing cases",
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} failing cases; first: {}",
        bad.len(),
        bad[0]
    );
}
