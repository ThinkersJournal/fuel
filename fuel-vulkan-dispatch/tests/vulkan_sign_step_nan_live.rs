// SPDX-License-Identifier: MIT OR Apache-2.0
//! `sign(NaN)` and `step(NaN)` through the SHIPPED Vulkan kernels on every non-CPU adapter (f32/f16/bf16/f64):
//! KISS 6.2-0009 pins them at ZERO (non-primitive ops follow their decomposition; torch differs and KISS wins:
//! pm-kernels 2026-11-26). CPU, Metal agree. Inputs `[NaN, 0.5, -2.0, 0.0]`: sign -> `[0, 1, -1, 0]`; step ->
//! `[0, 1, 0, _]`. An unbound op/dtype is reported `UNBOUND`, not failed.
//!
//! Gate: `require_gpu_run_lock`; a missing adapter is a FAILURE (GAP-157). Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-dispatch --test vulkan_sign_step_nan_live -- --nocapture --test-threads=1`.

use std::sync::{Arc, RwLock};

use fuel_dispatch::kernel::{KernelBindingTable, OpParams};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use fuel_vulkan_dispatch::vulkan_dispatch::register_vulkan_kernels;
use half::{bf16, f16};

const DTYPES: [DType; 4] = [DType::F32, DType::F16, DType::BF16, DType::F64];

fn enc(dt: DType, v: &[f32]) -> Vec<u8> {
    match dt {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F64 => v.iter().flat_map(|x| (*x as f64).to_le_bytes()).collect(),
        DType::F16 => v
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn dec(dt: DType, b: &[u8]) -> Vec<f32> {
    match dt {
        DType::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F64 => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        DType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        DType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        _ => unreachable!(),
    }
}

fn run(be: &Arc<VulkanBackend>, op: OpKind, dt: DType, input: &[f32]) -> Result<Vec<f32>, String> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let alts = table.lookup_alternatives(op, &[dt, dt], BackendId::Vulkan);
    let alt = alts.first().ok_or("UNBOUND")?;
    let h = be
        .upload_bytes_handle(&enc(dt, input))
        .map_err(|e| format!("{e:?}"))?;
    let src = Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(h), dt)));
    let out = be
        .alloc_bytes_handle(input.len() * dt.size_in_bytes())
        .map_err(|e| format!("{e:?}"))?;
    let dst = Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(out), dt)));
    let layouts = [
        Layout::contiguous(Shape::from_dims(&[input.len()])),
        Layout::contiguous(Shape::from_dims(&[input.len()])),
    ];
    (alt.kernel)(&[src], &mut [Arc::clone(&dst)], &layouts, &OpParams::None)
        .map_err(|e| format!("{e:?}"))?;
    let bytes = match &dst.read().unwrap().inner {
        BackendStorage::Vulkan(v) => be.download_bytes(v).map_err(|e| format!("{e:?}"))?,
        _ => return Err("not vulkan".into()),
    };
    Ok(dec(dt, &bytes))
}

/// `"op/dtype"` cases on which the shipped Vulkan kernels DIVERGE from KISS 6.2-0009 today (measured 2026-11-26 on the
/// Radeon 610M): `sign(NaN)` = 1.0 for f32/f16/bf16 (the Slang `sign` intrinsic; the f64 kernel returns 0). Pinned both ways:
/// the test reddens when the kernels are fixed (delete the entries then). The `.spv` are fishstringer-built (Slang).
const KNOWN_DIVERGENT: &[&str] = &["sign/F32", "sign/F16", "sign/BF16"];

#[test]
fn vulkan_sign_and_step_of_nan_are_zero() {
    fuel_test_support::require_gpu_run_lock();
    let listed = VulkanBackend::list_devices().expect("list_devices");
    let input = [f32::NAN, 0.5, -2.0, 0.0];
    let cases: [(&str, OpKind, &[Option<f32>]); 2] = [
        (
            "sign",
            OpKind::SignElementwise,
            &[Some(0.0), Some(1.0), Some(-1.0), Some(0.0)],
        ),
        (
            "step",
            OpKind::StepElementwise,
            &[Some(0.0), Some(1.0), Some(0.0), None],
        ),
    ];
    let mut adapters = 0;
    for (idx, name, kind) in &listed {
        if kind == "cpu" {
            continue;
        }
        let be = Arc::new(
            VulkanBackend::with_selection(DeviceSelection::Index(*idx))
                .unwrap_or_else(|e| panic!("open adapter {name:?}: {e:?}")),
        );
        let mut bad = Vec::new();
        for dt in DTYPES {
            for (op_name, op, want) in cases {
                match run(&be, op, dt, &input) {
                    Err(e) if e == "UNBOUND" => {
                        println!("VKSIGN {name} | {op_name}/{dt:?}: UNBOUND")
                    }
                    Err(e) => panic!("{name}/{op_name}/{dt:?}: dispatch failed: {e}"),
                    Ok(got) => {
                        let ok = got.iter().zip(want).all(|(g, w)| match w {
                            None => true,
                            Some(x) => !g.is_nan() && g == x,
                        });
                        println!(
                            "VKSIGN {name} | {op_name}/{dt:?}: got {got:?} {}",
                            if ok { "OK" } else { "DIVERGES" }
                        );
                        let key = format!("{op_name}/{dt:?}");
                        let known = KNOWN_DIVERGENT.contains(&key.as_str());
                        if known == ok {
                            bad.push(format!(
                                "{key}: got {got:?} (known divergent: {known}, agrees: {ok})"
                            ));
                        }
                    }
                }
            }
        }
        assert!(bad.is_empty(), "{name}: {bad:?}");
        adapters += 1;
    }
    assert!(adapters > 0, "no non-CPU Vulkan adapter found (GAP-157)");
}
