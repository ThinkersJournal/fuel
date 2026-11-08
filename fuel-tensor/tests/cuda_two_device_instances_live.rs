// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-359 CUDA arm: the FIRST realize in a process on a CUDA device failed with
//! `no backend supports write_slice on [F32, F32]; available backends: [Cpu, Cuda]`
//! (lightbulb, fuel 0.43.2; `decode_step_bench` prefill, then the served driver ended
//! after step 0 with no tokens).
//!
//! **Cause (measured 2026-11-08, 4070):** `build_optimized_graph` took its
//! `SystemTopology::current()` snapshot BEFORE `cpu_accel::ensure_installed()` installed
//! the CUDA kernels (row 137 step 6.2 moved that install above `fuel-dispatch`). The
//! topology is built from the binding table, so `backends_for(Cuda { gpu_id: 0 })` was
//! empty for that first realize (`placements=[]` printed at the planner's
//! empty-enumeration site) and the planner reported a missing binding that exists.
//! Later realizes saw the refreshed topology and worked, which made the failure depend on
//! which test (or call) ran first. Two wrong leads were refuted on the way: two
//! `CudaDevice` instances on one ordinal (distinct `DeviceId`s) and a CPU-resident
//! persistent buffer both PASS once the topology is right.
//!
//! The tests follow `KvCache::write_bytes` (a persistent device buffer seeded into the
//! executor's `StorageCache`, then a `write_slice` onto it). Whichever of them runs first in
//! the process is the born-red reproduction (`--test-threads=1` runs them alphabetically, so
//! `persistent_write_slice_cpu_resident_buffer_cuda_realize` is first and was red at the base).
//!
//! Compiled only under `--features cuda`; needs a live NVIDIA GPU (fails loudly without one). Run:
//!   cargo test -p fuel-tensor --features cuda --test cuda_two_device_instances_live -- --test-threads=1

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::CudaDevice;
use fuel_dispatch::pipelined::StorageCache;
use fuel_ir::{DType, HostBuffer, Shape};
use fuel_memory::Storage;
use fuel_tensor::Device;
use fuel_tensor::lazy::Tensor;

/// A live CUDA device is REQUIRED (GAP-157): this file is compiled only under
/// `--features cuda`, so a missing device is a loud failure, not a silent pass.
fn device() -> CudaDevice {
    let probe = fuel_hardware::probe::ProbeReport::probe_all();
    fuel_test_support::require(
        "a live CUDA device",
        probe
            .devices
            .iter()
            .any(|d| d.backend == fuel_ir::probe::BackendId::Cuda),
    );
    CudaDevice::new(0).expect("cuda device 0 should be available")
}

/// Allocate the persistent `[4, 2]` F32 buffer (all zeros) on `alloc_dev`, then
/// write `[7, 8]` into row 1 of it by realizing on `realize_dev`. Returns the
/// realized bytes, or the error text.
fn write_row_through(alloc_dev: &CudaDevice, realize_dev: &CudaDevice) -> Result<Vec<f32>, String> {
    write_row_via(&alloc_dev.clone().into(), &realize_dev.clone().into())
}

/// Same, with arbitrary allocation and realize `Device`s (so the buffer can be
/// CPU-resident while the realize is pinned to CUDA).
fn write_row_via(alloc: &Device, realize: &Device) -> Result<Vec<f32>, String> {
    let buf: Arc<RwLock<Storage>> = fuel_tensor::pipelined_bridge::upload_host_buffer_to_device(
        alloc,
        HostBuffer::F32(vec![0.0; 8]),
    )
    .map_err(|e| format!("allocating the persistent buffer: {e:?}"))?;

    let src = Tensor::from_f32(
        vec![7.0_f32, 8.0],
        Shape::from_dims(&[1, 2]),
        &Device::cpu(),
    )
    .map_err(|e| format!("building src: {e:?}"))?;
    let dest = src.const_placeholder_like(Shape::from_dims(&[4, 2]), DType::F32);
    let post = dest
        .write_slice(&src, vec![(1, 2), (0, 2)])
        .map_err(|e| format!("write_slice builds: {e:?}"))?;

    let mut cache = StorageCache::new();
    cache.insert(dest.node_id(), buf);
    let bytes = fuel_tensor::pipelined_bridge::realize_one_bytes_with_initial(
        post.graph_handle(),
        post.node_id(),
        realize,
        cache,
    )
    .map_err(|e| format!("realize on the second instance: {e:?}"))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

const WANT: [f32; 8] = [0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0];

#[test]
fn persistent_write_slice_same_instance() {
    let a = device();
    let got = write_row_through(&a, &a).expect("control: one instance end to end");
    assert_eq!(got, WANT.to_vec());
}

#[test]
fn persistent_write_slice_second_instance_same_ordinal() {
    let a = device();
    let b = device();
    assert!(
        !a.same_device(&b),
        "premise: two CudaDevice::new(0) are distinct Fuel devices (DeviceId differs)"
    );
    let got = write_row_through(&a, &b)
        .expect("a buffer allocated through one CudaDevice instance must be writable through another on the same ordinal");
    assert_eq!(got, WANT.to_vec());
}

/// A CPU-resident persistent buffer under a CUDA-pinned realize (one suspect on the way to
/// the cause; it passes and is kept as a regression guard for that shape).
#[test]
fn persistent_write_slice_cpu_resident_buffer_cuda_realize() {
    let a = device();
    let got = write_row_via(&Device::cpu(), &a.clone().into())
        .expect("a CPU-resident persistent buffer must be writable by a CUDA-pinned realize");
    assert_eq!(got, WANT.to_vec());
}
