// SPDX-License-Identifier: MIT OR Apache-2.0
// HOSTED-CI NOTE (PIP #715 N2): this test needs a live Vulkan adapter, so the hosted `rust-ci.yml` test steps skip it BY NAME
// (`--skip` list); a rename of any test here MUST be mirrored there, or hosted CI reddens. The GPU runner
// (`device-tests.yml`) runs it whole.
//! **What the SHIPPED Vulkan `.spv` argmax/argmin do on the NaN spec vectors** (the CPU rule since
//! the torch-convention ruling: a NaN WINS and the FIRST NaN's index is returned; no NaN: ties take
//! the first index, `-0.0 == +0.0`), for f32/f16/bf16/f64 on every non-CPU Vulkan adapter, named by
//! adapter name and PCI address. The union standard for a kernel family is that every backend agrees
//! on the spec vectors; this test MEASURES Vulkan against them.
//!
//! `KNOWN_DIVERGENT` pins today's divergences in both directions (a divergence not listed fails the
//! test; a listed one that now agrees fails it too), so the list cannot outlive the defect.
//!
//! Gate: `require_gpu_run_lock`; a missing adapter is a FAILURE (GAP-157). Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-dispatch --test vulkan_arg_nan_vectors_live -- --nocapture --test-threads=1`.

use std::sync::{Arc, RwLock};

use fuel_dispatch::kernel::{KernelBindingTable, OpParams};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::probe::device_identities;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use fuel_vulkan_dispatch::vulkan_dispatch::register_vulkan_kernels;
use half::{bf16, f16};

const DTYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F64];
const NAN: f64 = f64::NAN;

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

/// One argmax/argmin over the last dim of a `[1, n]` row; returns the index.
fn arg(be: &Arc<VulkanBackend>, op: OpKind, dt: DType, row: &[f64]) -> Result<u32, String> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let alts = table.lookup_alternatives(op, &[dt, DType::U32], BackendId::Vulkan);
    let alt = alts.first().ok_or("no kernel registered")?;
    let h = be
        .upload_bytes_handle(&enc(dt, row))
        .map_err(|e| format!("{e:?}"))?;
    let input = Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(h), dt)));
    let out = be.alloc_bytes_handle(4).map_err(|e| format!("{e:?}"))?;
    let oa = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Vulkan(out),
        DType::U32,
    )));
    let layouts = [
        Layout::contiguous(Shape::from_dims(&[1, row.len()])),
        Layout::contiguous(Shape::from_dims(&[1])),
    ];
    (alt.kernel)(
        &[input],
        &mut [Arc::clone(&oa)],
        &layouts,
        &OpParams::Reduce {
            dims: vec![1],
            keepdim: false,
        },
    )
    .map_err(|e| format!("{e:?}"))?;
    let bytes = match &oa.read().unwrap().inner {
        BackendStorage::Vulkan(v) => be.download_bytes(v).map_err(|e| format!("{e:?}"))?,
        _ => return Err("not vulkan".into()),
    };
    Ok(u32::from_le_bytes(bytes[..4].try_into().unwrap()))
}

/// `(name, row, want_argmax, want_argmin)` -- the spec vectors.
fn vectors() -> Vec<(String, Vec<f64>, u32, u32)> {
    let mut v: Vec<(String, Vec<f64>, u32, u32)> = vec![
        ("nan-mid".into(), vec![1.0, NAN, 0.0, 5.0], 1, 1),
        ("nan-last".into(), vec![1.0, 5.0, NAN], 2, 2),
        ("nan-first".into(), vec![NAN, 1.0, 5.0], 0, 0),
        ("two-nans".into(), vec![3.0, NAN, NAN, 1.0], 1, 1),
        ("all-nan".into(), vec![NAN, NAN, NAN], 0, 0),
        ("ties".into(), vec![5.0, 5.0, 2.0, 2.0], 0, 2),
        ("zeros-a".into(), vec![-0.0, 0.0, 0.0], 0, 0),
        ("zeros-b".into(), vec![0.0, -0.0, -0.0], 0, 0),
    ];
    // Longer rows (the workgroup-parallel path): a NaN deep in the row, no NaN with ties.
    let mut long: Vec<f64> = (0..1024).map(|i| (i % 7) as f64).collect();
    long[700] = NAN;
    v.push(("long-nan-700".into(), long, 700, 700));
    let ties: Vec<f64> = (0..1024).map(|i| (i % 7) as f64).collect();
    v.push(("long-ties".into(), ties, 6, 0));
    v
}

/// Vectors on which the shipped Vulkan `.spv` DIVERGES from the spec today (measured 2026-11-25 on
/// the AMD Radeon 610M): every vector that holds a NaN except the all-NaN row (index 0 either way).
/// The kernels IGNORE a NaN (comparisons against NaN are false), so e.g. `[1,NaN,0,5]` gives argmax 3,
/// argmin 2 where the spec says 1, 1. Matched by the vector-name suffix of `"op/dtype/vector"`.
const KNOWN_DIVERGENT: &[&str] = &[
    "/nan-mid",
    "/nan-last",
    "/nan-first",
    "/two-nans",
    "/long-nan-700",
];

#[test]
fn vulkan_arg_reduce_nan_spec_vectors() {
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
        println!("ARGNAN adapter: name={name:?} type={kind} pci={pci:?}");
        let be = Arc::new(
            VulkanBackend::with_selection(DeviceSelection::Index(*idx))
                .unwrap_or_else(|e| panic!("open adapter {name:?}: {e:?}")),
        );
        assert_eq!(
            &be.device_name, name,
            "opened adapter must be the listed one"
        );
        let (mut unexpected_div, mut unexpected_agree) = (Vec::new(), Vec::new());
        for dt in DTYPES {
            for (vname, row, want_max, want_min) in vectors() {
                for (op, opname, want) in [
                    (OpKind::ArgMaxDim, "argmax", want_max),
                    (OpKind::ArgMinDim, "argmin", want_min),
                ] {
                    if dt == DType::BF16 && row.len() % 2 == 1 {
                        // The bf16 kernel needs an even last dim (lane pair): a contract limit, not a NaN rule.
                        println!(
                            "ARGNAN {name} | {opname}/{dt:?}/{vname}: SKIP (odd bf16 last dim)"
                        );
                        continue;
                    }
                    let got = arg(&be, op, dt, &row).unwrap_or_else(|e| {
                        panic!("{opname}/{dt:?}/{vname}: dispatch failed: {e}")
                    });
                    let key = format!("{opname}/{dt:?}/{vname}");
                    let agree = got == want;
                    println!(
                        "ARGNAN {name} | {key}: got {got} want {want} {}",
                        if agree { "OK" } else { "DIVERGES" }
                    );
                    let known = KNOWN_DIVERGENT.iter().any(|k| key.ends_with(k));
                    match (known, agree) {
                        (false, false) => unexpected_div.push(key),
                        (true, true) => unexpected_agree.push(key),
                        _ => {}
                    }
                }
            }
        }
        assert!(
            unexpected_div.is_empty() && unexpected_agree.is_empty(),
            "{name}: arg-reduce NaN behaviour vs the spec. Unlisted divergences: {unexpected_div:?}. Listed but now agreeing: {unexpected_agree:?}"
        );
        ran += 1;
    }
    assert!(
        ran > 0,
        "no non-CPU Vulkan adapter found (GAP-157: absence is a failure, not a pass)"
    );
}
