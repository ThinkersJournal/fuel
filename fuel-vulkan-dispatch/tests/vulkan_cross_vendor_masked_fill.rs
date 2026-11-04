// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-vendor comparison for the Vulkan `masked_fill` claim (live, `#[ignore]`d). Moved here from the
//! unit-test module of `fuel-dispatch`'s `seed_vulkan_ledger` (ROADMAP row 137 step 6.1): it runs the
//! verification recipes against the Vulkan registrar, which `fuel-dispatch` can no longer name. It reaches
//! the harness through the hidden `fuel_dispatch::dev_tools::vulkan_verify_only_on`, which returns only the
//! adapter name, its ordinal and each attempt's op, dtypes and outcome, and writes nothing.

use fuel_dispatch::dev_tools;
use fuel_vulkan_dispatch::register_vulkan_kernels;

/// **Cross-vendor comparison for the `masked_fill` byte-select claim —
/// VERIFY ONLY, writes nothing.**
///
/// The contract asserts *"pure byte select — exact, no arithmetic,
/// bit-identical across any hardware."* Every record backing it was earned
/// on ONE adapter (`PreferDiscrete` → the discrete card), so it is evidence
/// about that driver, not about the kernel. This runs the same recipes on
/// the OTHER vendor and compares outcomes.
///
/// ⚠️ **It must never write the ledger.** `LedgerRecord` has no adapter
/// field and `upsert` keys on `(backend, dtypes, revision, claim)`, so
/// persisting these would REPLACE the discrete card's records in place —
/// same count, same keys, different hardware, no signal. The ledger cannot
/// represent a two-vendor claim; that is a schema question, not this test's.
///
/// ⚠️ **`force: true` is load-bearing.** Without it `has_pass` short-
/// circuits every key already recorded — i.e. exactly the six under test —
/// and the run would verify nothing while reporting success.
///
/// ⚠️ **The adapter is asserted, not requested.** `ByName("AMD")` is what we
/// ask for; `vendor_id == 0x1002` cross-checked against the backend's own
/// `gpu_id` is what we got. A silent fallback to the discrete card would
/// otherwise produce a same-vendor run that passes and reads as
/// cross-vendor.
///
/// Run:
///   `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-dispatch \
///    --test vulkan_cross_vendor_masked_fill -- --ignored --nocapture`
#[test]
#[ignore = "cross-vendor comparison: needs a second Vulkan adapter"]
fn masked_fill_cross_vendor_second_adapter_verify_only() {
    use fuel_vulkan_backend::DeviceSelection;

    // Run BOTH adapters through the SAME function with the SAME flags, so
    // the adapter is the only variable. Comparing an AMD run here against
    // the discrete card's run inside the *seeder* would differ in call site
    // as well as hardware — two variables, one conclusion.
    let run = |sel: DeviceSelection, want_vendor: u32, label: &str| {
        let (name, gpu_id, log) =
            match dev_tools::vulkan_verify_only_on(register_vulkan_kernels, sel, true) {
                Ok(v) => v,
                Err(e) => {
                    println!("[x-vendor] {label}: cannot run: {e:?}");
                    return None;
                }
            };
        let descs =
            fuel_vulkan_backend::probe::enumerate_devices().expect("vulkan probe enumerates");
        let d = descs
            .iter()
            .find(|d| d.device_index as usize == gpu_id)
            .unwrap_or_else(|| panic!("no descriptor for gpu_id {gpu_id}"))
            .clone();
        println!(
            "[x-vendor] {label}: gpu_id={gpu_id} name={name:?} sku={:?} vendor=0x{:04x}",
            d.hardware_sku, d.vendor_id,
        );
        // The adapter is ASSERTED, never inferred from what we requested.
        assert_eq!(
            d.vendor_id, want_vendor,
            "{label}: wanted vendor {want_vendor:#06x}, got {:#06x} ({}). A silent \
             fallback would make this a same-vendor run that still passes.",
            d.vendor_id, d.hardware_sku,
        );
        let mf: Vec<(Vec<fuel_ir::DType>, String)> = log
            .iter()
            .filter(|a| a.op == "MaskedFill")
            .map(|a| (a.dtypes.clone(), a.outcome.clone()))
            .collect();
        for (dt, out) in &mf {
            println!("[x-vendor] {label}: MaskedFill {dt:?}: {out}");
        }
        Some((d, mf))
    };

    let Some((amd_dev, amd)) = run(DeviceSelection::ByName("AMD".to_string()), 0x1002, "AMD")
    else {
        println!("[x-vendor] second adapter unavailable — reporting and stopping.");
        return;
    };
    let Some((nv_dev, nv)) = run(DeviceSelection::PreferDiscrete, 0x10de, "NVIDIA") else {
        println!("[x-vendor] discrete adapter unavailable — cannot compare.");
        return;
    };

    assert!(
        !amd.is_empty() && !nv.is_empty(),
        "a comparison over zero MaskedFill attempts is vacuous — `force` did not take effect"
    );

    // Same recipes, same flags: the ATTEMPT SETS must match, or the two
    // sides are not comparable and a "0 failing" summary would be counting
    // different populations.
    let key = |v: &Vec<(Vec<fuel_ir::DType>, String)>| {
        let mut k: Vec<String> = v.iter().map(|(d, _)| format!("{d:?}")).collect();
        k.sort();
        k
    };
    assert_eq!(
        key(&amd),
        key(&nv),
        "⚠️ the two adapters attempted DIFFERENT dtype sets, so their outcomes are not \
         comparable. AMD={:?} NVIDIA={:?}",
        key(&amd),
        key(&nv),
    );

    let fails = |v: &Vec<(Vec<fuel_ir::DType>, String)>| -> Vec<_> {
        v.iter()
            .filter(|(_, o)| o.contains("fail"))
            .cloned()
            .collect()
    };
    println!(
        "[x-vendor] {} attempts each — {} ({} failing) vs {} ({} failing)",
        amd.len(),
        amd_dev.hardware_sku,
        fails(&amd).len(),
        nv_dev.hardware_sku,
        fails(&nv).len(),
    );
    assert!(
        fails(&amd).is_empty() && fails(&nv).is_empty(),
        "⚠️ DIVERGENCE: masked_fill is contract-claimed 'bit-identical across any \
         hardware'. AMD failures={:?}; NVIDIA failures={:?}. The CONTRACT'S CLAIM is \
         then wrong, which outranks any ledger record — report, do not reconcile.",
        fails(&amd),
        fails(&nv),
    );
}
