// SPDX-License-Identifier: MIT OR Apache-2.0
//! The snapshot hazard for the Vulkan kernels (ROADMAP row 137 step 6.1).
//!
//! `fuel-dispatch` no longer registers the Vulkan kernels; `fuel-vulkan-dispatch` adds them to the global
//! binding table through an extension, and `fuel-tensor` triggers that in `cpu_accel::ensure_installed`.
//! Readers take a SNAPSHOT of the table; an extension publishes a new table for later readers and leaves a
//! snapshot already taken untouched. So a Vulkan kernel installed after a realize's snapshot is invisible to
//! that realize, with no error. These tests fail if the install is removed, or runs after the snapshot.
//!
//! This file is its own test binary, so the global table starts untouched by any other test.

#![cfg(feature = "vulkan")]

use fuel_dispatch::dispatch::global_bindings;
use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::{Shape, probe::BackendId};
use fuel_tensor::lazy::Tensor;

fn vulkan_keys(table: &KernelBindingTable) -> usize {
    table
        .iter_keys()
        .filter(|(_, _, backend)| *backend == BackendId::Vulkan)
        .count()
}

/// Behaviour: a snapshot taken before the first lazy realize has no Vulkan kernels (that IS the hazard, and
/// the positive control); a plain CPU realize through the production bridge installs them; a snapshot
/// taken afterwards has them; the old snapshot is still empty.
#[test]
fn the_first_realize_installs_vulkan_and_a_stale_snapshot_never_sees_it() {
    let stale = global_bindings();
    assert_eq!(
        vulkan_keys(&stale),
        0,
        "positive control failed: the global table already held Vulkan kernels before any realize, so this \
         test cannot tell 'installed by the realize' from 'was always there'"
    );

    let a = Tensor::from_f32(
        vec![1.0_f32, 2.0],
        Shape::from_dims(&[2]),
        &fuel_tensor::Device::cpu(),
    )
    .expect("CPU tensor");
    assert_eq!(a.add(&a).expect("add").realize_f32(), vec![2.0, 4.0]);

    let fresh = global_bindings();
    assert!(
        vulkan_keys(&fresh) > 0,
        "the bridge realize did not install the Vulkan kernels: `cpu_accel::ensure_installed` must call \
         `fuel_vulkan_dispatch::install_global`"
    );
    assert_eq!(
        vulkan_keys(&stale),
        0,
        "a snapshot is immutable: it must not gain the later extension"
    );
}

/// Order: in the production optimize entry (`build_optimized_graph`), `ensure_installed()` must come BEFORE
/// both snapshots the optimizer reads: the `SystemTopology` (built from the binding table, so a backend
/// installed after it is missing from `backends_for(..)`; GAP-359: the first CUDA realize failed with
/// `NoBackendForOp`) and the binding-table snapshot. The behaviour test cannot see that order (it only sees
/// the table after the realize), so this one reads the source. It fails if the install moves after either
/// snapshot or is removed; if `pipelined_bridge.rs` is restructured, update the anchors here AND re-prove
/// the order.
#[test]
fn the_bridge_installs_before_it_takes_the_optimize_snapshots() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pipelined_bridge.rs");
    let src = std::fs::read_to_string(path).expect("read pipelined_bridge.rs");
    let lines: Vec<&str> = src.lines().map(str::trim).collect();

    let start = lines
        .iter()
        .position(|l| l.starts_with("fn build_optimized_graph("))
        .expect("positive control failed: `fn build_optimized_graph(` is gone or renamed; find the new entry");
    // The function body ends at the next top-level item, which starts at column 0 in the source.
    let end = src
        .lines()
        .enumerate()
        .skip(start + 1)
        .find(|(_, l)| {
            l.starts_with("fn ") || l.starts_with("pub fn ") || l.starts_with("pub(crate) fn ")
        })
        .map_or(lines.len(), |(i, _)| i);
    let body = &lines[start..end];
    let find = |needle: &str| body.iter().position(|l| *l == needle);

    let install = find("crate::cpu_accel::ensure_installed();")
        .expect("`ensure_installed()` is missing from build_optimized_graph: Vulkan and CUDA kernels would be invisible");
    let topology = find("let topology = SystemTopology::current();")
        .expect("positive control failed: the topology snapshot line is gone or renamed; re-prove the order");
    let bindings = find("let bindings_guard = global_bindings();")
        .expect("positive control failed: the binding-table snapshot line is gone or renamed; re-prove the order");

    assert!(
        install < topology,
        "`ensure_installed()` (body line {install}) must run BEFORE the SystemTopology snapshot (line {topology}):          a backend installed after it is absent from `backends_for(..)` and the first realize in a process fails          with NoBackendForOp (GAP-359)"
    );
    assert!(
        install < bindings,
        "`ensure_installed()` (body line {install}) must run BEFORE the binding-table snapshot (line {bindings}),          or kernels registered later are invisible to this realize"
    );
}
