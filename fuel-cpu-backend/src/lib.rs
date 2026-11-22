// SPDX-License-Identifier: MIT OR Apache-2.0
//! CPU backend computation kernels for fuel.
//!
//! This crate contains the core CPU computation helpers (MatMul, pooling,
//! convolution, reductions, etc.) extracted from `fuel-core` so they can be
//! reused and tested independently. It also re-exports the Accelerate FFI bindings
//! when the `accelerate` feature is enabled.

#[cfg(feature = "accelerate")]
pub mod accelerate;

// Link glue. `accelerate` declares `extern "C"` FFI against the vendor library;
// nothing in this crate NAMES the `-src` crate that supplies the link directives,
// and rustc drops an unreferenced crate from the link line, so `extern crate` is
// what forces the native library in. It lives HERE, beside the FFI it serves, so a
// binary that enables only `fuel-cpu-backend/accelerate` links too (GAP-348 (b);
// it used to sit in `fuel-core/src/lib.rs`).
#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub mod conv2d;
pub mod dyn_impl;
pub mod host_storage;
pub mod ops;
pub mod probe;
pub mod quantized;
pub mod system_memory;
pub mod utils;

/// Phase 7.5 storage-unification target: byte-shaped CPU storage
/// with `Arc<AlignedBytes>` backing, 64-byte alignment, and CoW
/// mutation. Coexists with the legacy `CpuStorage` (HostBuffer-based)
/// during op-kernel migration; legacy retires when the last kernel
/// migrates.
pub mod byte_storage;

/// Typed byte-shaped kernels — Phase 7.5 B5. These operate on
/// `CpuStorageBytes` directly via `bytemuck` typed views. The
/// dispatch wrapper in `fuel_dispatch::dispatch::cpu_wrappers`
/// extracts `CpuStorageBytes` from `BackendStorage::Cpu(...)` and
/// calls these kernels.
pub mod byte_kernels;

/// Trait-chassis surface shared by elementwise + reduction CPU
/// kernels. One shape/loop pass per kernel family; op-specific math
/// lives in tiny trait impls. See [`chassis::reduction`].
pub mod chassis;

pub use byte_storage::{CPU_ALIGN_BYTES, CpuStorageBytes};
pub use dyn_impl::CpuStorage;
pub use ops::*;
pub use quantized::CpuQStorage;
pub use utils::*;
