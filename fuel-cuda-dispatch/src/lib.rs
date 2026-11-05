// SPDX-License-Identifier: MIT OR Apache-2.0
//! CUDA kernel bindings for [`fuel_dispatch`] (ROADMAP row 137, step 6.2).
//!
//! These lived inside `fuel-dispatch` behind `feature = "cuda"`; they now sit **above** it, the way
//! `fuel-vulkan-dispatch` (step 6.1) and the vendor CPU backends do: they use the public kernel-table API
//! of `fuel-dispatch` and hand it a registrar. `fuel-dispatch` no longer registers the baracuda kernels, so
//! **something has to call [`install_global`]** before the process-wide binding table is read for a plan.
//! `fuel-tensor` does it (`cpu_accel::ensure_installed`, which runs right before the bridge takes its
//! snapshot; see the snapshot hazard in `fuel-vulkan-dispatch`'s crate docs, which applies unchanged).
//! A caller that builds its own `KernelBindingTable` calls [`register_cuda_kernels`] on it instead.
//!
//! The whole crate body is behind the `cuda` feature (compiling it runs the baracuda forge).
//!
//! # Layout
//!
//! - `baracuda_dispatch`: the baracuda-backed wrappers and `register_baracuda_cuda_kernels` (moved
//!   unchanged; the module keeps its name so the paths inside it and in `link` did not change).
//! - `link`: the FKC link registry that binds contract entry points to those wrappers.

#![cfg(feature = "cuda")]

pub mod baracuda_dispatch;
pub mod link;

pub use baracuda_dispatch::register_baracuda_cuda_kernels;
pub use link::CudaLinkRegistry;

use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::probe::BackendId;
use std::sync::Once;

/// Register every CUDA kernel on `table`: the PTX-path registrations that stay in `fuel-dispatch`
/// (`dispatch::register_cuda_kernels`), then the baracuda wrappers, then the bulk cost fill for CUDA
/// entries still on the unknown-cost sentinel (the pass `register_optional_backends` used to run).
pub fn register_cuda_kernels(table: &mut KernelBindingTable) {
    fuel_dispatch::dispatch::register_cuda_kernels(table);
    register_baracuda_cuda_kernels(table);
    table.fill_unset_cost_for_backend(
        BackendId::Cuda,
        fuel_dispatch::cost::default_cost_for_op_kind,
    );
}

/// Add the CUDA kernels to the process-wide binding table, once per process, together with the derived
/// `BackendCapabilities` for the CUDA backend. Safe to call from any thread any number of times.
pub fn install_global() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        fuel_dispatch::dispatch::extend_global_bindings(|table| {
            register_cuda_kernels(table);
            fuel_dispatch::dispatch::register_derived_caps_for(BackendId::Cuda, table);
        });
    });
}
