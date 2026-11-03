// SPDX-License-Identifier: MIT OR Apache-2.0
//! Prints the Vulkan adapter inventory this box actually exposes.
//!
//! WHY THIS EXISTS. `VulkanBackend::new()` uses `DeviceSelection::PreferDiscrete`
//! and hands back the discrete card. CLAUDE.md records the consequence as a
//! standing trap: a test that means to cover two vendors "silently becomes a
//! same-vendor test that still passes". Nothing in a green live-GPU run says
//! WHICH adapter produced it, so every such green is reported against an
//! assumed device.
//!
//! `PreferDiscrete should give NVIDIA` is an INFERENCE. The name in this
//! output is a MEASUREMENT. This test exists so a report can cite the second.
//!
//! It asserts almost nothing on purpose — it is an instrument, not a gate. The
//! one thing it does assert is that enumeration returned at least one adapter,
//! so that an empty inventory fails loudly instead of printing nothing and
//! passing (a live-GPU test that cannot run must say so, never report `ok`).

#[test]
#[ignore = "live GPU: enumerates real Vulkan adapters; run via scripts/gpu-run.ps1"]
fn print_the_adapter_inventory() {
    let listed = fuel_vulkan_backend::VulkanBackend::list_devices()
        .expect("list_devices failed on a box that is supposed to have Vulkan");

    println!("=== VULKAN ADAPTER INVENTORY (measured) ===");
    for (idx, name, kind) in &listed {
        println!("  [{idx}] {kind:<10} {name}");
    }

    let descriptors =
        fuel_vulkan_backend::probe::enumerate_devices().expect("probe enumeration failed");
    println!("=== PROBE DESCRIPTORS (vendor/device identity) ===");
    for d in &descriptors {
        println!(
            "  {} vendor_id=0x{:04x} device_id=0x{:04x}",
            d.hardware_sku, d.vendor_id, d.device_id
        );
    }

    // Which adapter `PreferDiscrete` yields is DERIVABLE from this inventory —
    // the selector takes the FIRST `discrete` and stops — but only when the
    // count of discrete adapters is unambiguous. Print the count so a reader
    // can see whether the derivation is sound rather than assume it.
    let discrete: Vec<&String> = listed
        .iter()
        .filter(|(_, _, k)| k == "discrete")
        .map(|(_, n, _)| n)
        .collect();
    println!(
        "=== discrete adapters: {} {:?} ===",
        discrete.len(),
        discrete
    );
    println!(
        "=== PreferDiscrete therefore selects: {} ===",
        match discrete.len() {
            1 => discrete[0].clone(),
            0 => "<none discrete; falls back to first non-CPU/OTHER>".to_string(),
            _ => "<AMBIGUOUS: more than one discrete adapter>".to_string(),
        }
    );

    assert!(
        !listed.is_empty(),
        "no Vulkan adapters enumerated — this test cannot report an inventory, \
         and must fail rather than pass having printed nothing"
    );

    every_reporting_adapter_reopens_by_its_identity();
}

/// `DeviceSelection::Identity` against the real adapters (ROADMAP row 137 step 4a). Called from the
/// ignored test above rather than being a test of its own: the `#[ignore]` count is a ratchet
/// (`fuel-ir/tests/suppression_ratchet.rs`), and this needs the same live device.
///
/// For every adapter that reports an identity, selecting by that identity must open the adapter at
/// the same enumeration ordinal and report an identity that is the same physical device. No
/// identity-reporting adapter FAILS (`required`): a live test that cannot run must not report `ok`.
fn every_reporting_adapter_reopens_by_its_identity() {
    use fuel_test_support::{required, required_ok};
    use fuel_vulkan_backend::{DeviceSelection, VulkanBackend, probe};

    let identities = required_ok("device_identities()", probe::device_identities());
    let reporting: Vec<(usize, _)> = identities
        .iter()
        .enumerate()
        .filter_map(|(i, id)| id.map(|id| (i, id)))
        .collect();
    required(
        "at least one Vulkan adapter that reports an identity",
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
        println!("=== identity selection: ordinal {ordinal} -> {got:?} ===");
    }
}
