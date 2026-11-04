// SPDX-License-Identifier: MIT OR Apache-2.0
//! `install_global` puts the Vulkan kernels and the derived Vulkan capabilities into the process-wide
//! tables exactly once (ROADMAP row 137 step 6.1). Hermetic: registration is pure table population and opens
//! no device. This file is its own test binary, so the global table starts untouched by any other test.

use fuel_dispatch::dispatch::{global_bindings, global_registry};
use fuel_dispatch::kernel::KernelBindingTable;
use fuel_ir::probe::BackendId;

fn vulkan_keys(table: &KernelBindingTable) -> usize {
    table
        .iter_keys()
        .filter(|(_, _, backend)| *backend == BackendId::Vulkan)
        .count()
}

fn vulkan_caps() -> usize {
    global_registry()
        .backends()
        .iter()
        .filter(|caps| caps.backend_id == BackendId::Vulkan)
        .count()
}

#[test]
fn install_global_adds_the_vulkan_kernels_and_capabilities_once_and_is_idempotent() {
    // `fuel-dispatch` no longer registers Vulkan itself: before the install the table has none. This is
    // the positive control for everything below (a table that already held Vulkan would make "added" vacuous).
    let before = global_bindings();
    assert_eq!(
        vulkan_keys(&before),
        0,
        "fuel-dispatch must not register Vulkan kernels on its own any more"
    );
    assert_eq!(vulkan_caps(), 0);

    fuel_vulkan_dispatch::install_global();
    let after = global_bindings();
    let n = vulkan_keys(&after);
    assert!(n > 0, "install_global registered no Vulkan kernels");
    assert_eq!(
        vulkan_caps(),
        1,
        "the derived Vulkan capabilities are registered exactly once"
    );

    // Idempotent: a second call neither panics (a duplicate registration would) nor adds anything.
    fuel_vulkan_dispatch::install_global();
    assert_eq!(vulkan_keys(&global_bindings()), n);
    assert_eq!(vulkan_caps(), 1);

    // A snapshot taken before the install keeps the old table: the extension is not retroactive.
    assert_eq!(vulkan_keys(&before), 0);
}
