// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-398: dropping a `VulkanBackend` with an OPEN (recorded, never submitted) batch must not crash.
//!
//! A realize that errors part-way returns with dispatches recorded into the recorder's open command buffer
//! and nothing submitted. `VulkanBackend` had no `Drop`, so the device was destroyed with that batch open and
//! the NVIDIA driver called a null pointer (exit `0xC0000005`, 70 of 70 runs of the row 124 probe before
//! row 119). `impl Drop for VulkanBackend` now flushes first. This test needs no failing realize: it records
//! three wired `QMatMul` (Q8_0 gemv) dispatches through the kernel wrapper WITHOUT a download or a
//! synchronize, checks that the batch is open (so the test is not vacuous), drops every handle, and returns.
//! RED = the test binary aborts at the drop (the exit code is the observation); GREEN = it passes.
//!
//! Risk, stated: at drop the flush submits the recorded work and waits on its fence. On an already-lost
//! device Vulkan specifies `VK_ERROR_DEVICE_LOST` from the wait (logged by the drop, no hang); that is the
//! specified behaviour and was NOT exercised here (no way was found to lose the device on purpose).
//!
//! Gate: the machine-wide GPU lock (`require_gpu_run_lock`); a missing Vulkan device is a FAILURE (GAP-157).
//! Not `#[ignore]`d. Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-dispatch \
//!  --test vulkan_backend_drop_open_batch_live`.

use std::sync::{Arc, RwLock};

use fuel_dispatch::kernel::{KernelBindingTable, OpParams};
use fuel_graph::QuantType;
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_quantized::{BlockQ8_0, GgmlType};
use fuel_vulkan_backend::VulkanBackend;
use fuel_vulkan_dispatch::vulkan_dispatch::register_vulkan_kernels;

#[test]
fn dropping_a_backend_with_an_open_batch_does_not_crash() {
    fuel_test_support::require_gpu_run_lock();
    let backend = Arc::new(
        VulkanBackend::new().expect("a Vulkan device (a missing adapter is a failure, GAP-157)"),
    );
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let kernel = table.lookup_alternatives(
        OpKind::QMatMul,
        &[DType::F32, DType::U32, DType::F32],
        BackendId::Vulkan,
    )[0]
    .kernel;

    let (m, k, n) = (1usize, 256usize, 16usize);
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32 + 1.0) * 0.009).cos() * 0.35)
        .collect();
    let mut w_blocks = vec![BlockQ8_0::zeros(); n * (k / BlockQ8_0::BLCK_SIZE)];
    BlockQ8_0::from_float(&w, &mut w_blocks);
    // SAFETY: a block is a `#[repr(C)]` plain-old-data struct; this reads its bytes.
    let w_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            w_blocks.as_ptr() as *const u8,
            std::mem::size_of_val(w_blocks.as_slice()),
        )
    }
    .to_vec();
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.021).sin() * 0.7).collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();

    let a_arc = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Vulkan(backend.upload_bytes_handle(&a_bytes).expect("upload a")),
        DType::F32,
    )));
    let w_arc = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Vulkan(backend.upload_bytes_handle(&w_bytes).expect("upload w")),
        DType::U32,
    )));
    let layouts = [
        Layout::contiguous(Shape::from_dims(&[m, k])),
        Layout::contiguous(Shape::from_dims(&[w_bytes.len() / 4])),
        Layout::contiguous(Shape::from_dims(&[m, n])),
    ];
    let mut outs = Vec::new();
    for _ in 0..3 {
        let out = Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(backend.alloc_bytes_handle(m * n * 4).expect("alloc")),
            DType::F32,
        )));
        kernel(
            &[Arc::clone(&a_arc), Arc::clone(&w_arc)],
            &mut [Arc::clone(&out)],
            &layouts,
            &OpParams::QMatMul {
                quant_type: QuantType::Q8_0,
                batch_count: 1,
                m,
                n,
                k,
            },
        )
        .expect("qmatmul Q8_0");
        outs.push(out);
    }
    // Control: the dispatches are recorded and NOT submitted, so the drop below is a drop with an open batch.
    let open = backend.open_batch_len();
    println!("open batch before drop: {open} dispatches");
    assert!(open >= 3, "the batch must be open at drop: {open}");

    // Drop every handle that carries an `Arc<VulkanBackend>`, then the backend itself.
    drop(outs);
    drop(a_arc);
    drop(w_arc);
    assert_eq!(
        Arc::strong_count(&backend),
        1,
        "something still holds the backend, so the drop below would not be the last"
    );
    drop(backend);
    println!("backend dropped");
}
