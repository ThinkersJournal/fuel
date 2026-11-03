// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backend-contract traits for the fuel ML framework.
//!
//! This crate holds the object-safe trait surface that **every backend
//! implements**. It is the PARKED trait-object alternative to the closed
//! `fuel_memory::BackendStorage` enum that the executor matches (see
//! "Status" below). It sits **above [`fuel_ir`]** (the vocabulary
//! crate it borrows `DType`/`Shape`/`Layout`/`HostBuffer`/the capability
//! data-types from) and **below the backend crates** (`fuel-cpu-backend`,
//! `fuel-cuda-backend`, `fuel-vulkan-backend`, `fuel-metal-backend`, …),
//! which depend on it to implement the traits.
//!
//! ## What lives here
//!
//! - [`dyn_backend`] — [`DynBackendStorage`] / [`DynBackendDevice`], the
//!   object-safe storage + device traits (parked, see "Status" below).
//! - [`backend`] — the capability/runtime traits [`HostStorage`],
//!   [`BackendStorage`], [`BackendCapabilityProvider`], [`BackendRuntime`],
//!   and the Tier-2 [`BackendStreams`] live-load extension.
//!   (Their *data* types — `SubstrateClass`, `TransferPath`,
//!   `BackendCapabilities`, `FitStatus` — stay in [`fuel_ir::backend`].)
//! - [`quantized`] — [`DynQuantizedStorage`] / [`QuantizedDeviceKernels`].
//!   (The `GgmlDType` *data* tag stays in [`fuel_ir::quantized`].)
//! - [`inplace_op`] — [`InplaceOp1`] / [`InplaceOp2`] / [`InplaceOp3`].
//! - [`storage`] — [`Storage`], the type-erased `Box<dyn DynBackendStorage>`
//!   handle that carries a backend's storage, plus `allocate_bundled_storage`.
//!   (The bundle *data* — `OutputView`, `OutputViewSpec`, `compose_bundle` —
//!   stays in [`fuel_ir::storage`].)
//!
//! ## Status: parked (ROADMAP row 137 step 5, 2026-11-03)
//!
//! The executor does NOT dispatch through [`DynBackendStorage`] / [`DynBackendDevice`]. It matches the
//! closed `fuel_memory::BackendStorage` enum and calls the methods defined on that enum, such as
//! `backend_id`, `location` and `copy_from` (CireSnave's ruling of 2026-11-01, S3 design note section 7). These traits stay as they are: the CPU, CUDA, Metal
//! and Vulkan backends implement them (`dyn_impl.rs`, `quantized.rs`), they are not extended, and
//! whether they are deleted is a separate decision.
//!
//! Measured at `origin/main` 043de464 (0.37.18): outside this crate and the four backend crates,
//! `DynBackendStorage` appears five times, all in comments (`fuel-dispatch/src/pipelined.rs`,
//! `fuel-graph/src/lib.rs`, `fuel-ir/src/{lib,storage,backend}.rs`); no code in `fuel-dispatch`,
//! `fuel-tensor`, `fuel-graph` or `fuel-ir` names it. CodeRipper 0.20.1 `reachability` run on this
//! crate alone reports `BackendCapabilityProvider`, `InplaceOp1`, `InplaceOp2` and `InplaceOp3` as
//! having zero callers inside it, and does not report `DynBackendStorage` (it has callers in this
//! crate). That check is per crate, so it says nothing about callers in the backend crates; a
//! cross-crate caller count is what a deletion decision needs.

pub mod backend;
pub mod dyn_backend;
pub mod inplace_op;
pub mod quantized;
pub mod storage;

pub use backend::{
    BackendCapabilityProvider, BackendRuntime, BackendStorage, BackendStreams, HostStorage,
};
pub use dyn_backend::{DynBackendDevice, DynBackendStorage};
pub use inplace_op::{InplaceOp1, InplaceOp2, InplaceOp3};
pub use quantized::{DynQuantizedStorage, QuantizedDeviceKernels};
pub use storage::{Storage, allocate_bundled_storage};
