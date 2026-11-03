// SPDX-License-Identifier: MIT OR Apache-2.0
//! Census of the Vulkan binding table: the oracle for ROADMAP row 137 step 6.1 and for any later change to
//! what `register_vulkan_kernels` registers (hermetic: pure table population, no device).
//!
//! The census is every Vulkan entry as `op dtypes backend source generic revision-hash`, sorted. The kernel
//! function pointer is deliberately NOT part of it (not stable, and not the claim). The move from
//! `fuel-dispatch` was proved by dumping this list at the base commit and at the moved tree and diffing the
//! files (identical); `golden/vulkan_binding_census.txt` keeps that list so a later change that adds or drops
//! a registration shows up as a diff here. To regenerate it after an intentional change:
//! `FUEL_CENSUS_OUT=fuel-vulkan-dispatch/tests/golden/vulkan_binding_census.txt cargo test -p
//! fuel-vulkan-dispatch --test binding_census`.

use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::probe::BackendId;
use fuel_vulkan_dispatch::register_vulkan_kernels;

fn census() -> Vec<String> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let mut lines: Vec<String> = table
        .iter_entries()
        .filter(|(_, _, backend, _)| *backend == BackendId::Vulkan)
        .map(|(op, dtypes, backend, entry)| {
            format!(
                "{op:?} {dtypes:?} {backend:?} src={} generic={} rev={:016x}",
                entry.kernel_source, entry.is_generic, entry.kernel_revision_hash
            )
        })
        .collect();
    lines.sort();
    lines
}

#[test]
fn the_vulkan_binding_table_matches_the_golden_census() {
    let got = census().join("\n");
    if let Some(path) = std::env::var_os("FUEL_CENSUS_OUT") {
        std::fs::write(&path, format!("{got}\n")).expect("write the census");
        return;
    }
    let golden = include_str!("golden/vulkan_binding_census.txt");
    // Compare line by line so a failure names the first difference instead of dumping two walls of text.
    let want: Vec<&str> = golden.lines().collect();
    let have: Vec<&str> = got.lines().collect();
    assert_eq!(
        have.len(),
        want.len(),
        "the Vulkan registrations changed in COUNT: golden {}, now {}",
        want.len(),
        have.len()
    );
    for (i, (h, w)) in have.iter().zip(&want).enumerate() {
        assert_eq!(h, w, "census line {i} differs");
    }
}

/// The census is not vacuous: it has entries, and the three kernel families that matter most are in it.
#[test]
fn the_census_is_not_empty_and_covers_matmul_softmax_and_binary_add() {
    let lines = census();
    assert!(lines.len() > 100, "only {} Vulkan entries", lines.len());
    for needle in ["MatMul", "SoftmaxLastDim", "AddElementwise"] {
        assert!(
            lines.iter().any(|l| l.starts_with(needle)),
            "no {needle} entry in the Vulkan census"
        );
    }
}
