// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backend-contract traits for the fuel ML framework.
//!
//! This crate holds the object-safe trait surface that **every backend
//! implements**. It is the trait-object alternative to the closed
//! `fuel_memory::BackendStorage` enum that the executor matches; it is still
//! live at the graph's host-data boundary (see "Status" below). It sits **above [`fuel_ir`]** (the vocabulary
//! crate it borrows `DType`/`Shape`/`Layout`/`HostBuffer`/the capability
//! data-types from) and **below the backend crates** (`fuel-cpu-backend`,
//! `fuel-cuda-backend`, `fuel-vulkan-backend`, `fuel-metal-backend`, …),
//! which depend on it to implement the traits.
//!
//! ## What lives here
//!
//! - [`dyn_backend`] — [`DynBackendStorage`] / [`DynBackendDevice`], the
//!   object-safe storage + device traits (kept, see "Status" below).
//! - [`backend`] — the capability/runtime traits [`HostStorage`],
//!   [`BackendStorage`], [`BackendCapabilityProvider`], [`BackendRuntime`],
//!   and the Tier-2 [`BackendStreams`] live-load extension.
//!   (Their *data* types — `SubstrateClass`, `TransferPath`,
//!   `BackendCapabilities`, `FitStatus` — stay in [`fuel_ir::backend`].)
//! - [`quantized`] — [`DynQuantizedStorage`] / [`QuantizedDeviceKernels`].
//!   (The `GgmlDType` *data* tag stays in [`fuel_ir::quantized`].)
//! - [`storage`] — [`Storage`], the type-erased `Box<dyn DynBackendStorage>`
//!   handle that carries a backend's storage, plus `allocate_bundled_storage`.
//!   (The bundle *data* — `OutputView`, `OutputViewSpec`, `compose_bundle` —
//!   stays in [`fuel_ir::storage`].)
//!
//! ## Status: kept, not extended (ROADMAP row 137 step 5, 2026-11-03)
//!
//! The executor does NOT dispatch through [`DynBackendStorage`] / [`DynBackendDevice`]: it matches the
//! closed `fuel_memory::BackendStorage` enum and calls the methods defined on that enum, such as
//! `backend_id`, `location` and `copy_from` (CireSnave's ruling of 2026-11-01, S3 design note section 7).
//! The traits are NOT dead, though, and were not deleted. `fuel-graph` builds a tensor from host data
//! through `DynBackendDevice::storage_from_host_buffer_owned_dyn`, which returns a
//! `Box<dyn DynBackendStorage>`, and wraps it with [`Storage::from_dyn`] (`NodeHandle::from_*`,
//! `const_like`, `build_filled_const`: 3 production call sites in `fuel-graph/src/lib.rs`, plus one in its
//! test module; 7 more uses of `allocate_bundled_storage`, production and tests not separated; `fuel-tensor` and `fuel` import the [`Storage`] handle). The four backends
//! implement the traits. They are not extended; removing them means re-plumbing that construction path.
//!
//! Measured at `origin/main` 043de464 (0.37.18), code only (comments stripped): the name
//! `DynBackendStorage` appears in this crate and the four backend crates and nowhere else; the five other
//! mentions are comments. Dependants: crates.io lists 12 reverse dependencies of this crate (all Fuel
//! workspace members); the one outside project on this machine that depends on it, Lightbulb, uses only
//! `backend::BackendRuntime`. `InplaceOp1/2/3` were removed in 0.38.0 (no
//! implementation, no caller). [`BackendCapabilityProvider`] is kept although no backend implements it and no
//! code calls it: that is known and deliberate, because the architecture document
//! (`docs/architecture/05-backend-contract.md`, Tier 1) specifies it and the dlpack extension spec depends on it.

pub mod backend;
pub mod dyn_backend;
pub mod quantized;
pub mod storage;

pub use backend::{
    BackendCapabilityProvider, BackendRuntime, BackendStorage, BackendStreams, HostStorage,
};
pub use dyn_backend::{DynBackendDevice, DynBackendStorage};
pub use quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
pub use storage::{Storage, allocate_bundled_storage};
