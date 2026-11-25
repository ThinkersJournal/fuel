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

/// Ties and NaN: CHARACTERISATION of baracuda's comparator at `0.1.0-alpha.186`
/// (recorded by the first live run on an RTX 4070, 2026-11-25), pinned so a baracuda
/// change reddens this test instead of silently moving the CUDA path. It DIVERGES
/// from the CPU kernel (and from `torch.topk`) in two ways, both documented in the
/// CUDA contract and GAP-402:
///
/// * ties: values agree, INDICES do not take the lower-index-first rule
///   (input `[3,5,5,1,5,2,4,5]`, k = 3: CUDA picks indices `[1, 2, 7]`, CPU `[1, 2, 4]`);
/// * NaN: a NaN in the row CORRUPTS the selection (input `[1, NaN, 9, 2]`, k = 2: CUDA
///   returns `([2.0, NaN], [3, 1])` and loses the 9.0; CPU returns `([NaN, 9.0], [1, 2])`).
///   A NaN-bearing row must not be sent to the CUDA kernel.
#[test]
fn cuda_topk_f32_ties_and_nan_diverge_from_the_cpu_kernel_at_alpha_186() {
    let x = [3.0_f32, 5.0, 5.0, 1.0, 5.0, 2.0, 4.0, 5.0];
    let got = run_cuda_f32(&x, 1, 8, 3);
    let want = run_cpu_f32(&x, 1, 8, 3);
    assert_eq!(got.0, want.0, "tie VALUES agree");
    assert_eq!(
        want.1,
        vec![1, 2, 4],
        "the CPU kernel takes the lowest tied indices"
    );
    assert_eq!(
        got.1,
        vec![1, 2, 7],
        "baracuda alpha.186 tie indices (recorded)"
    );
    let nanx = [1.0_f32, f32::NAN, 9.0, 2.0];
    let got = run_cuda_f32(&nanx, 1, 4, 2);
    let want = run_cpu_f32(&nanx, 1, 4, 2);
    assert!(want.0[0].is_nan() && want.0[1] == 9.0 && want.1 == vec![1, 2]);
    assert!(got.0[0] == 2.0 && got.0[1].is_nan(), "recorded: {got:?}");
    assert_eq!(got.1, vec![3, 1], "recorded: the 9.0 at index 2 is lost");
}
