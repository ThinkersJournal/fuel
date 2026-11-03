// SPDX-License-Identifier: MIT OR Apache-2.0
//! The topology reports Vulkan once this crate's registrar has run.
//!
//! Moved from `fuel-dispatch/src/topology.rs` (row 137 step 6.1): `SystemTopology` learns a backend from
//! the kernels registered in the global binding table, and Vulkan's kernels are registered by
//! [`fuel_vulkan_dispatch::install_global`], which `fuel-dispatch` can no longer call (it sits below this
//! crate). In `fuel-dispatch`'s own test binary the registrar never runs, so that test would be red; here
//! it runs first, which is the production order. No device is needed: the binding table is what is read.

use fuel_dispatch::topology::SystemTopology;
use fuel_ir::DeviceLocation;
use fuel_ir::probe::BackendId;

#[test]
fn vulkan_backend_discovered_after_install_global() {
    fuel_vulkan_dispatch::install_global();
    let topology = SystemTopology::current();
    assert!(
        topology.backends().contains(&BackendId::Vulkan),
        "install_global ran; topology should report Vulkan. Got: {:?}",
        topology.backends(),
    );
    let vk_dev = DeviceLocation::Vulkan { gpu_id: 0 };
    let here = topology.backends_for(vk_dev);
    assert!(
        here.contains(&BackendId::Vulkan),
        "Vulkan should be among the backends targeting {vk_dev:?}, got {here:?}",
    );
}

/// Sabotage sibling: without the registrar, a fresh process sees no Vulkan. Runs in its own test binary
/// section only if it executes before `install_global` in the same process, so it checks the
/// registrar-free table directly instead of racing the other test.
#[test]
fn an_empty_binding_table_does_not_report_vulkan() {
    let table = fuel_dispatch::kernel::KernelBindingTable::new();
    assert!(
        !table
            .iter_entries()
            .any(|(_, _, b, _)| b == BackendId::Vulkan),
        "a table nobody registered into must not contain Vulkan entries",
    );
    let mut with = fuel_dispatch::kernel::KernelBindingTable::new();
    fuel_vulkan_dispatch::vulkan_dispatch::register_vulkan_kernels(&mut with);
    assert!(
        with.iter_entries()
            .any(|(_, _, b, _)| b == BackendId::Vulkan),
        "control: the registrar must put Vulkan entries into a table",
    );
}
