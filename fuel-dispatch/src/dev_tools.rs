// SPDX-License-Identifier: MIT OR Apache-2.0
//! Maintenance tools behind the `dev-tools` feature. **Not part of the
//! supported API; semver-exempt.**
//!
//! These were `#[ignore]`d tests: each one REWRITES a committed file (the
//! verified ledger, or the CPU contracts), so they are tools, not assertions.
//! A test that is ignored on purpose reads as flaky or slow and cannot say what
//! it really is. They live here so the `fuel-dev-tools` binary (a separate
//! crate, unable to reach `pub(crate)` internals) can drive them, and so
//! `cargo check --features dev-tools --bins` keeps them compiling.

use crate::fkc::{contract_audit_flip, verify};

/// Re-seed the CPU half of `fuel-dispatch/kernel-contracts/fkc-verified-ledger.json`.
pub fn seed_cpu_ledger() {
    verify::seed_cpu_ledger::run_seed_cpu_ledger();
}

/// Re-seed the CUDA half of the verified ledger. Needs a live CUDA device. `register` is the baracuda
/// registrar (`fuel_cuda_dispatch::register_baracuda_cuda_kernels`); `fuel-dispatch` cannot name it, so the
/// `fuel-cuda-dev-tools` binary passes it in.
#[cfg(feature = "cuda")]
pub fn seed_cuda_ledger(register: fn(&mut crate::kernel::KernelBindingTable)) {
    verify::seed_cuda_ledger::run_seed_cuda_ledger(register);
}

/// Earn the CUDA f32 matmul's `max_relative`/`max_absolute` ledger records (ROADMAP row 39).
/// Needs a live CUDA device.
#[cfg(feature = "cuda")]
pub fn seed_cuda_accuracy(register: fn(&mut crate::kernel::KernelBindingTable)) {
    verify::seed_cuda_accuracy::run_seed_cuda_accuracy(register);
}

/// Measure the CUDA f32 accuracy claim and print; writes nothing (ROADMAP row 39). Needs a live
/// CUDA device.
#[cfg(feature = "cuda")]
pub fn check_cuda_accuracy(register: fn(&mut crate::kernel::KernelBindingTable)) {
    verify::seed_cuda_accuracy::run_check_cuda_accuracy(register);
}

/// The CUDA accuracy claims, one row per claimed op: `(contract file stem under kernel-contracts/cuda, kernel
/// section name, max_relative, max_absolute, op, number of dtypes of the F32 entry)`. The test that keeps
/// these in step with the contracts moved to `fuel-cuda-dispatch` with the contracts (row 137 step 6.2).
#[cfg(feature = "cuda")]
pub fn cuda_accuracy_claim_rows() -> Vec<(
    &'static str,
    &'static str,
    f64,
    f64,
    fuel_ir::dispatch::OpKind,
    usize,
)> {
    verify::seed_cuda_accuracy::claim_rows()
        .into_iter()
        .map(|(stem, k, c, op, n)| (stem, k, c.max_relative, c.max_absolute, op, n))
        .collect()
}

/// The CUDA f32 `qmatmul` claim `(max_relative, max_absolute)` (see [`cuda_accuracy_claim_rows`]).
#[cfg(feature = "cuda")]
pub fn cuda_qmatmul_f32_claim() -> (f64, f64) {
    let c = verify::seed_cuda_accuracy::QMATMUL_F32_CLAIM;
    (c.max_relative, c.max_absolute)
}

/// Re-seed the Vulkan half of the verified ledger. Needs a live Vulkan device. `register` is the Vulkan
/// registrar (`fuel_vulkan_dispatch::register_vulkan_kernels`); `fuel-dispatch` cannot name it, so the
/// `fuel-vulkan-dev-tools` binary passes it in.
#[cfg(feature = "vulkan")]
pub fn seed_vulkan_ledger(register: fn(&mut crate::kernel::KernelBindingTable)) {
    verify::seed_vulkan_ledger::run_seed_vulkan_ledger(register);
}

/// The Vulkan f32 accuracy claims the seeder earns, as `(kernel, max_relative, max_absolute)`. The contracts that
/// declare the same numbers moved to `fuel-vulkan-dispatch`, so the test that keeps the two in step moved with
/// them and reads the claims through this function (the constants themselves are private to `fkc::verify`).
#[cfg(feature = "vulkan")]
pub fn vulkan_f32_accuracy_claims() -> [(&'static str, f64, f64); 4] {
    use verify::seed_vulkan_matmul_accuracy as s;
    [
        (
            "matmul_f32",
            s::MATMUL_F32_CLAIM.max_relative,
            s::MATMUL_F32_CLAIM.max_absolute,
        ),
        (
            "rms_norm_last_dim_f32",
            s::RMS_NORM_F32_CLAIM.max_relative,
            s::RMS_NORM_F32_CLAIM.max_absolute,
        ),
        (
            "softmax_last_dim_f32",
            s::SOFTMAX_F32_CLAIM.max_relative,
            s::SOFTMAX_F32_CLAIM.max_absolute,
        ),
        (
            "rope_f32",
            s::ROPE_F32_CLAIM.max_relative,
            s::ROPE_F32_CLAIM.max_absolute,
        ),
    ]
}

/// One Vulkan verification attempt, as the cross-vendor comparison reads it (the harness's own record type is
/// private to `fkc::verify`).
#[cfg(feature = "vulkan")]
pub struct VulkanVerifyAttempt {
    pub op: String,
    pub dtypes: Vec<fuel_ir::DType>,
    pub outcome: String,
}

/// Run the Vulkan verification recipes on ONE adapter and return `(adapter name, gpu_id, attempts)`. VERIFY
/// ONLY: it writes nothing (the ledger cannot represent a two-vendor claim). `force` re-verifies keys the
/// ledger already holds. Needs a live Vulkan device; `register` is the Vulkan registrar.
#[cfg(feature = "vulkan")]
pub fn vulkan_verify_only_on(
    register: fn(&mut crate::kernel::KernelBindingTable),
    selection: fuel_vulkan_backend::DeviceSelection,
    force: bool,
) -> Result<(String, usize, Vec<VulkanVerifyAttempt>), String> {
    let (name, gpu_id, _ledger, log) =
        verify::seed_vulkan_ledger::run_vulkan_verification_on(register, selection, force)
            .map_err(|e| format!("{e:?}"))?;
    let attempts = log
        .into_iter()
        .map(|a| VulkanVerifyAttempt {
            op: a.op,
            dtypes: a.dtypes,
            outcome: a.outcome,
        })
        .collect();
    Ok((name, gpu_id, attempts))
}

/// Earn the Vulkan f32 matmul's `max_relative`/`max_absolute` ledger records (board 240).
/// Needs a live Vulkan device.
#[cfg(feature = "vulkan")]
pub fn seed_vulkan_matmul_accuracy(register: fn(&mut crate::kernel::KernelBindingTable)) {
    verify::seed_vulkan_matmul_accuracy::run_seed_vulkan_matmul_accuracy(register);
}

/// Measure the Vulkan f32 accuracy claims and print; writes nothing (board 240). Needs a live
/// Vulkan device; `FUEL_VULKAN_TEST_ADAPTER` selects the adapter.
#[cfg(feature = "vulkan")]
pub fn check_vulkan_accuracy(register: fn(&mut crate::kernel::KernelBindingTable)) {
    verify::seed_vulkan_matmul_accuracy::run_check_vulkan_accuracy(register);
}

/// GAP-228: flip the fully-backed `audited: false` CPU contract sections.
pub fn gap_228_flip() {
    contract_audit_flip::run_gap_228_flip();
}

/// GAP-228: report which sections are wholesale flippable (read-only).
pub fn gap_228_sections_report() {
    contract_audit_flip::run_gap_228_sections_report();
}
