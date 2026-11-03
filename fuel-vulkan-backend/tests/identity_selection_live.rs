// SPDX-License-Identifier: MIT OR Apache-2.0
//! `DeviceSelection::Identity` against the real adapters (ROADMAP row 137 step 4a).
//!
//! For every Vulkan device that reports an identity, selecting by that identity must open the
//! device at the same enumeration ordinal and report an identity that is the same physical device.
//! A box with no identity-reporting device FAILS (`required`), never reports `ok`: a live test that
//! cannot run must say so.
//!
//! Run via `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-backend --test
//! identity_selection_live -- --ignored`. This crate has no `vulkan` feature; it IS the backend.

use fuel_test_support::{required, required_ok};
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend, probe};

#[test]
#[ignore = "live GPU: opens each real Vulkan adapter by identity; run via scripts/gpu-run.ps1"]
fn selecting_by_identity_opens_the_device_at_that_ordinal() {
    let identities = required_ok("device_identities()", probe::device_identities());
    let reporting: Vec<(usize, _)> = identities
        .iter()
        .enumerate()
        .filter_map(|(i, id)| id.map(|id| (i, id)))
        .collect();
    required(
        "at least one Vulkan device that reports an identity",
        (!reporting.is_empty()).then_some(()),
    );
    for (ordinal, wanted) in reporting {
        let backend = required_ok(
            "VulkanBackend::with_selection(Identity)",
            VulkanBackend::with_selection(DeviceSelection::Identity(wanted)),
        );
        assert_eq!(
            backend.gpu_id, ordinal,
            "identity {wanted:?} resolved to a different ordinal than the enumeration order"
        );
        let got = required("the opened backend reports an identity", backend.identity());
        assert!(
            got.same_physical_device(&wanted),
            "opened {got:?}, asked for {wanted:?}"
        );
        println!("identity-selection: ordinal {ordinal} -> {got:?}");
    }
}

#[test]
#[ignore = "live GPU: asks for an identity no device has; run via scripts/gpu-run.ps1"]
fn an_unknown_identity_is_an_error_not_a_fallback() {
    let nobody = fuel_ir::probe::DeviceIdentity {
        uuid: Some([0xAB; 16]),
        pci: None,
    };
    // `VulkanBackend` is not `Debug`, so match instead of `unwrap_err`.
    match VulkanBackend::with_selection(DeviceSelection::Identity(nobody)) {
        Ok(b) => panic!("opened ordinal {} for an identity no device has", b.gpu_id),
        Err(e) => assert!(
            e.to_string().contains("no Vulkan device has identity"),
            "{e}"
        ),
    }
}
