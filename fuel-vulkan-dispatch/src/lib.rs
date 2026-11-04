// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vulkan kernel bindings for [`fuel_dispatch`] (ROADMAP row 137, step 6.1).
//!
//! These lived inside `fuel-dispatch` behind `feature = "vulkan"`; they now sit **above** it, the way the
//! vendor CPU backends (`fuel-aocl-cpu-backend`, `fuel-mkl-cpu-backend`) already do: they use the public
//! kernel-table API of `fuel-dispatch` and hand it a registrar. `fuel-dispatch` no longer registers
//! Vulkan kernels itself, so **something has to call [`install_global`]** before the process-wide binding
//! table is read for a plan. `fuel-tensor` does it (its `cpu_accel::ensure_installed`, which runs right
//! before the bridge takes its snapshot, see the hazard below). A caller that builds its own
//! `KernelBindingTable` calls [`register_vulkan_kernels`] on it instead.
//!
//! # The snapshot hazard
//!
//! Readers of the global table take a snapshot (`fuel_dispatch::dispatch::global_bindings`). An
//! extension (`extend_global_bindings`) publishes a new table for LATER readers; a snapshot already
//! taken keeps the old one. So a Vulkan kernel registered after a snapshot is invisible to whoever holds
//! that snapshot, with no error. The rule: install before the first snapshot that a Vulkan-capable
//! realize will use. `fuel-tensor/tests/vulkan_install_before_snapshot.rs` fails if that order breaks.
//!
//! # Layout
//!
//! - [`vulkan_dispatch`]: the Vulkan kernel wrappers and [`register_vulkan_kernels`] (moved unchanged
//!   from `fuel-dispatch/src/vulkan_dispatch.rs`; the module keeps its name so the paths inside it and
//!   in [`link`] did not change).
//! - [`link`]: the FKC link registry that binds contract entry points to those wrappers.

pub mod link;
pub mod vulkan_dispatch;

pub use link::{VULKAN_CAST_ENTRY_POINTS, VULKAN_ELEMENTWISE_ENTRY_POINTS, VulkanLinkRegistry};
pub use vulkan_dispatch::register_vulkan_kernels;

use std::sync::Once;

/// Add the Vulkan kernels to the process-wide binding table, once per process, together with the
/// derived `BackendCapabilities` for the Vulkan backend (the step `fuel-dispatch` used to run for every
/// compiled-in GPU backend right after registering them). Safe to call from any number of threads and
/// any number of times: later calls return immediately. See the crate docs for when it must run.
pub fn install_global() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        fuel_dispatch::dispatch::extend_global_bindings(|table| {
            register_vulkan_kernels(table);
            fuel_dispatch::dispatch::register_derived_caps_for(
                fuel_ir::probe::BackendId::Vulkan,
                table,
            );
        });
    });
}
