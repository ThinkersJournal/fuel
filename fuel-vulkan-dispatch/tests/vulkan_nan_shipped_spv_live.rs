// SPDX-License-Identifier: MIT OR Apache-2.0
//! **What the SHIPPED Vulkan `.spv` does with a NaN operand** in unary `relu`, the value reduces
//! `max`/`min` and the last-dim reduces `max`/`min`, for f32/f16/bf16/f64, on every non-CPU Vulkan
//! adapter on the box, named by adapter name and PCI address (never by enumeration index).
//!
//! Fuel's convention (KISS-Ops 6.11-0002, torch parity): `relu`, `Maximum`, `Minimum` and the reduces
//! PROPAGATE NaN. The binary `Maximum`/`Minimum` kernels have an explicit `isnan` guard and are the
//! POSITIVE CONTROL run through the same dispatch path: if they do not propagate, the harness (not
//! the reduces) is broken.
//!
//! # Known-failing set, pinned in both directions
//!
//! `KNOWN_DROP_GROUPS` lists every kernel group that DROPS the NaN today. A case that drops and is
//! not covered fails the test, and a covered case that now propagates fails it too, so the list
//! cannot outlive the defect: when the guards land (fishstringer), this test reddens and the entries
//! are deleted (and GAP-381 closed). A kernel that errors is a failure, not a drop.
//!
//! Gate: the machine-wide GPU lock (`require_gpu_run_lock`), and a missing adapter is a FAILURE
//! (GAP-157). Not `#[ignore]`d; run on the GPU runner with
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-dispatch --test vulkan_nan_shipped_spv_live -- --nocapture --test-threads=1`.

use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use fuel_dispatch::kernel::{KernelBindingTable, OpParams};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::probe::device_identities;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use fuel_vulkan_dispatch::vulkan_dispatch::register_vulkan_kernels;
use half::{bf16, f16};

const DTYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F64];

fn enc(dt: DType, v: &[f64]) -> Vec<u8> {
    match dt {
        DType::F32 => v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect(),
        DType::F64 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => v
            .iter()
            .flat_map(|x| f16::from_f64(*x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f64(*x).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn dec(dt: DType, b: &[u8]) -> Vec<f64> {
    match dt {
        DType::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c) as f64)
            .collect(),
        DType::F64 => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect(),
        DType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f64())
            .collect(),
        DType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f64())
            .collect(),
        _ => unreachable!(),
    }
}

/// Dispatch one kernel through the binding table. `inputs` are `(data, dims)`.
fn run(
    be: &Arc<VulkanBackend>,
    op: OpKind,
    dt: DType,
    inputs: &[(&[f64], &[usize])],
    out_dims: &[usize],
    params: OpParams,
) -> Result<Vec<f64>, String> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let mut key = vec![dt; inputs.len()];
    key.push(dt);
    let alts = table.lookup_alternatives(op, &key, BackendId::Vulkan);
    let alt = alts.first().ok_or("no kernel registered")?;
    let mut ins = Vec::new();
    let mut layouts = Vec::new();
    for (data, dims) in inputs {
        let h = be
            .upload_bytes_handle(&enc(dt, data))
            .map_err(|e| format!("{e:?}"))?;
        ins.push(Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(h),
            dt,
        ))));
        layouts.push(Layout::contiguous(Shape::from_dims(dims)));
    }
    layouts.push(Layout::contiguous(Shape::from_dims(out_dims)));
    let out_n: usize = out_dims.iter().product::<usize>().max(1);
    let out = be
        .alloc_bytes_handle(out_n * dt.size_in_bytes())
        .map_err(|e| format!("{e:?}"))?;
    let oa = Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(out), dt)));
    (alt.kernel)(&ins, &mut [Arc::clone(&oa)], &layouts, &params).map_err(|e| format!("{e:?}"))?;
    let bytes = match &oa.read().unwrap().inner {
        BackendStorage::Vulkan(v) => be.download_bytes(v).map_err(|e| format!("{e:?}"))?,
        _ => return Err("not vulkan".into()),
    };
    Ok(dec(dt, &bytes))
}

/// Every case on one adapter: `(key, propagated)`. A dispatch error panics (a failure, not a drop).
fn measure(be: &Arc<VulkanBackend>) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut put = |key: String, r: Result<Vec<f64>, String>, idx: usize| {
        let v = r.unwrap_or_else(|e| panic!("{key}: dispatch failed: {e}"));
        out.push((key, v[idx].is_nan()));
    };
    for dt in DTYPES {
        // POSITIVE CONTROL: binary Maximum/Minimum carry an isnan guard and must propagate.
        for (op, n) in [
            (OpKind::MaximumElementwise, "maximum"),
            (OpKind::MinimumElementwise, "minimum"),
        ] {
            let r = run(
                be,
                op,
                dt,
                &[(&[f64::NAN, 1.0], &[2]), (&[1.0, f64::NAN], &[2])],
                &[2],
                OpParams::None,
            );
            put(format!("control/{n}-a/{dt:?}"), r.clone(), 0);
            put(format!("control/{n}-b/{dt:?}"), r, 1);
        }
        let r = run(
            be,
            OpKind::ReluElementwise,
            dt,
            &[(&[f64::NAN, -1.0, 2.0, f64::NAN], &[4])],
            &[4],
            OpParams::None,
        );
        put(format!("relu/elem0/{dt:?}"), r.clone(), 0);
        put(format!("relu/elem3/{dt:?}"), r, 3);
        for (op, name) in [(OpKind::MaxReduce, "max"), (OpKind::MinReduce, "min")] {
            for (n, nan_at, tag) in [
                (6usize, 3usize, "small"),
                (4096, 2000, "large-mid"),
                (4096, 0, "large-first"),
            ] {
                let mut d: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
                d[nan_at] = f64::NAN;
                let r = run(
                    be,
                    op,
                    dt,
                    &[(&d, &[n])],
                    &[1],
                    OpParams::Reduce {
                        dims: vec![0],
                        keepdim: false,
                    },
                );
                put(format!("{name}-full/{tag}/{dt:?}"), r, 0);
            }
            for (last, nan_at, tag) in [
                (8usize, 5usize, "short"),
                (256, 37, "lane37"),
                (1024, 700, "1024"),
            ] {
                let rows = 2;
                // Row 0 has ONE NaN; row 1 is clean and must keep its finite extreme (harness control).
                let mut d: Vec<f64> = (0..rows * last).map(|i| ((i % 5) as f64) + 1.0).collect();
                d[nan_at] = f64::NAN;
                let r = run(
                    be,
                    op,
                    dt,
                    &[(&d, &[rows, last])],
                    &[rows],
                    OpParams::Reduce {
                        dims: vec![1],
                        keepdim: false,
                    },
                );
                let v = r.clone().unwrap();
                let clean = if name == "max" { 5.0 } else { 1.0 };
                assert_eq!(
                    v[1], clean,
                    "{name}-lastdim/{tag}/{dt:?}: the clean row must keep its extreme"
                );
                put(format!("{name}-lastdim/{tag}/{dt:?}"), r, 0);
            }
        }
    }
    out
}

/// Kernel groups that DROP the NaN on the shipped `.spv` (every case, every dtype). Measured
/// 2026-11-10 at `de12d8efc` (spv built from fishstringer-kernels-source 0.0.13).
const KNOWN_DROP_GROUPS: &[&str] = &[
    "relu/",
    "max-full/",
    "min-full/",
    "max-lastdim/",
    "min-lastdim/",
];

#[test]
fn shipped_vulkan_spv_nan_behaviour_is_pinned() {
    fuel_test_support::require_gpu_run_lock();
    let listed = VulkanBackend::list_devices().expect("list_devices");
    let identities = device_identities().expect("device_identities");
    let mut ran = 0;
    for (idx, name, kind) in &listed {
        if kind == "cpu" {
            continue;
        }
        let pci = identities
            .get(*idx)
            .and_then(|i| i.as_ref())
            .and_then(|i| i.pci);
        println!("NANSPV adapter: name={name:?} type={kind} pci={pci:?}");
        let be = Arc::new(
            VulkanBackend::with_selection(DeviceSelection::Index(*idx))
                .unwrap_or_else(|e| panic!("open adapter {name:?}: {e:?}")),
        );
        assert_eq!(
            &be.device_name, name,
            "opened adapter must be the listed one"
        );
        let cases = measure(&be);
        let mut unexpected_drop = BTreeSet::new();
        let mut unexpected_propagate = BTreeSet::new();
        for (key, propagated) in &cases {
            println!(
                "NANSPV {name} | {key}: {}",
                if *propagated { "NAN-OUT" } else { "NO-NAN" }
            );
            if key.starts_with("control/") {
                assert!(
                    *propagated,
                    "{name}: POSITIVE CONTROL {key} did not propagate NaN; the harness is broken"
                );
                continue;
            }
            let known_drop = KNOWN_DROP_GROUPS.iter().any(|g| key.starts_with(g));
            match (known_drop, *propagated) {
                (true, true) => {
                    unexpected_propagate.insert(key.clone());
                }
                (false, false) => {
                    unexpected_drop.insert(key.clone());
                }
                _ => {}
            }
        }
        assert!(
            unexpected_drop.is_empty() && unexpected_propagate.is_empty(),
            "{name}: NaN behaviour changed. Now propagating (delete from KNOWN_DROP_GROUPS, close GAP-381): \
             {unexpected_propagate:?}. Newly dropping: {unexpected_drop:?}"
        );
        ran += 1;
    }
    assert!(
        ran > 0,
        "no non-CPU Vulkan adapter found (GAP-157: absence is a failure, not a pass)"
    );
}
