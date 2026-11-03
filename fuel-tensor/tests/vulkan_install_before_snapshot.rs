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

/// Order: in the production optimize entry, `ensure_installed()` must come right BEFORE the line that takes
/// the snapshot the optimizer reads. The behaviour test cannot see that order (it only sees the table after
/// the realize), so this one reads the source. It fails if the install moves after the snapshot or is
/// removed; if `pipelined_bridge.rs` is restructured, update the anchor here AND re-prove the order.
#[test]
fn the_bridge_installs_right_before_it_takes_the_optimize_snapshot() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pipelined_bridge.rs");
    let src = std::fs::read_to_string(path).expect("read pipelined_bridge.rs");
    let lines: Vec<&str> = src.lines().map(str::trim).collect();
    let snapshots: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| **l == "let bindings_guard = global_bindings();")
        .map(|(i, _)| i)
        .collect();
    assert!(
        !snapshots.is_empty(),
        "positive control failed: the optimize-entry snapshot line is gone or renamed; find the new one and \
         re-prove that the install still precedes it"
    );
    for i in snapshots {
        let window = &lines[i.saturating_sub(3)..i];
        assert!(
            window.contains(&"crate::cpu_accel::ensure_installed();"),
            "line {}: `ensure_installed()` must run just before this snapshot, or Vulkan kernels registered \
             later are invisible to this realize",
            i + 1
        );
    }
}
