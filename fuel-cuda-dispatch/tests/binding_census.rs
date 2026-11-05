// SPDX-License-Identifier: MIT OR Apache-2.0
//! Census of the CUDA binding table: the oracle for ROADMAP row 137 step 6.2 and for any later change to
//! what `register_cuda_kernels` registers (hermetic: pure table population, no device).
//!
//! The census is every CUDA entry as `op dtypes backend source generic revision-hash`, sorted. The kernel
//! function pointer is deliberately NOT part of it (not stable, and not the claim). The move from
//! `fuel-dispatch` was proved by dumping this list at the base commit (`2247bfb23`, built with
//! `--features cuda`: 476 entries) and at the moved tree and diffing the files (identical);
//! `golden/cuda_binding_census.txt` keeps that list so a later change that adds or drops a registration
//! shows up as a diff here. To regenerate it after an intentional change:
//! `FUEL_CENSUS_OUT=fuel-cuda-dispatch/tests/golden/cuda_binding_census.txt cargo test -p
//! fuel-cuda-dispatch --features cuda --test binding_census`.

#![cfg(feature = "cuda")]

use fuel_cuda_dispatch::register_cuda_kernels;
use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::probe::BackendId;

fn census_of(table: &KernelBindingTable) -> Vec<String> {
    let mut lines: Vec<String> = table
        .iter_entries()
        .filter(|(_, _, backend, _)| *backend == BackendId::Cuda)
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

fn census() -> Vec<String> {
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    census_of(&table)
}

/// First difference between two censuses, or `None` when identical (count first, then line by line, so a
/// failure names the first difference instead of dumping two walls of text).
fn first_difference(have: &[String], want: &[&str]) -> Option<String> {
    if have.len() != want.len() {
        return Some(format!(
            "COUNT differs: golden {}, now {}",
            want.len(),
            have.len()
        ));
    }
    have.iter()
        .zip(want)
        .enumerate()
        .find(|(_, (h, w))| h.as_str() != **w)
        .map(|(i, (h, w))| format!("line {i} differs: golden {w:?}, now {h:?}"))
}

#[test]
fn the_cuda_binding_table_matches_the_golden_census() {
    let got = census().join("\n");
    if let Some(path) = std::env::var_os("FUEL_CENSUS_OUT") {
        std::fs::write(&path, format!("{got}\n")).expect("write the census");
        return;
    }
    let golden = include_str!("golden/cuda_binding_census.txt");
    let want: Vec<&str> = golden.lines().collect();
    let have: Vec<String> = got.lines().map(str::to_owned).collect();
    if let Some(d) = first_difference(&have, &want) {
        panic!("the CUDA registrations changed: {d}");
    }
}

/// The census is not vacuous: it has entries, and the kernel families that matter most are in it.
#[test]
fn the_census_is_not_empty_and_covers_matmul_softmax_and_binary_add() {
    let lines = census();
    assert!(lines.len() > 100, "only {} CUDA entries", lines.len());
    for needle in ["MatMul", "SoftmaxLastDim", "AddElementwise"] {
        assert!(
            lines.iter().any(|l| l.starts_with(needle)),
            "no {needle} entry in the CUDA census"
        );
    }
}

/// Retained sabotage sibling: the comparison must see a dropped registration and a changed one. Without
/// this, a `first_difference` that always said "identical" would leave the golden test green for the
/// wrong reason.
#[test]
fn the_comparison_sees_a_dropped_and_a_changed_registration() {
    let golden = include_str!("golden/cuda_binding_census.txt");
    let want: Vec<&str> = golden.lines().collect();
    let mut have: Vec<String> = want.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(first_difference(&have, &want), None, "control: identical");
    let dropped = have.remove(7);
    let d = first_difference(&have, &want).expect("a dropped entry is seen");
    assert!(d.starts_with("COUNT differs"), "{d}");
    have.insert(7, format!("{dropped} MUTATED"));
    let d = first_difference(&have, &want).expect("a changed entry is seen");
    assert!(d.starts_with("line 7 differs"), "{d}");
}
