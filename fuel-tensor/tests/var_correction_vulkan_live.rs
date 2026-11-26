// SPDX-License-Identifier: MIT OR Apache-2.0
// HOSTED-CI NOTE (PIP #715 N2): this test needs a live Vulkan adapter, so the hosted `rust-ci.yml` test steps skip it BY NAME
// (`--skip` list); a rename of any test here MUST be mirrored there, or hosted CI reddens. The GPU runner
// (`device-tests.yml`) runs it whole.
//! `Tensor::var`'s divisor clamp (`max(0, n - correction)`, the torch rule; ruling 2026-11-25) run
//! through the Vulkan backend on every non-CPU adapter, f32. `var` is a RECIPE over
//! mean/sub/sqr/sum/mul_scalar, so Vulkan inherits the rule through its own kernels; this MEASURES it
//! (the CPU is measured by `phase_a3_keepdim_tests::var_divisor_is_clamped_at_zero_in_every_dtype`).
//! `x * inf` with `x == 0` must be NaN and with `x > 0` must be inf: a kernel that flushes or
//! folds that would show here.
//!
//! **Limit of this measurement:** every COMPUTED node of the recipe is stamped to the adapter and the
//! graph is realized through `realize_f32_vulkan`; there is no counter that proves each kernel ran on
//! the GPU (no Vulkan-only divergent op sits in the recipe to act as a positive control), so the claim
//! is "the Vulkan realize path of the placed graph gives these values".
//!
//! Gate: `require_gpu_run_lock`; a missing adapter is a FAILURE (GAP-157). Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-tensor --features vulkan --test var_correction_vulkan_live -- --nocapture --test-threads=1`.

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use fuel_ir::Shape;
use fuel_tensor::lazy::Tensor;
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

/// `(name, data, shape, dim, correction, want)`; `want`: NaN / +inf / a finite value.
type VarCase = (&'static str, Vec<f32>, Vec<usize>, usize, f64, f32);

fn vectors() -> Vec<VarCase> {
    vec![
        ("one-elem-c1", vec![5.0], vec![1], 0, 1.0, f32::NAN),
        (
            "two-rows-of-one-c1",
            vec![5.0, 7.0],
            vec![2, 1],
            1,
            1.0,
            f32::NAN,
        ),
        (
            "n2-c2-nonzero-sum",
            vec![1.0, 3.0],
            vec![2],
            0,
            2.0,
            f32::INFINITY,
        ),
        ("one-elem-c2", vec![5.0], vec![1], 0, 2.0, f32::NAN),
        ("n2-c3", vec![1.0, 3.0], vec![2], 0, 3.0, f32::INFINITY),
        ("one-elem-c0", vec![5.0], vec![1], 0, 0.0, 0.0),
        ("n2-c1-control", vec![1.0, 3.0], vec![2], 0, 1.0, 2.0),
    ]
}

#[test]
fn var_divisor_clamp_on_every_vulkan_adapter() {
    fuel_test_support::require_gpu_run_lock();
    let listed = VulkanBackend::list_devices().expect("list_devices");
    let mut ran = 0;
    for (idx, name, kind) in &listed {
        if kind == "cpu" {
            continue;
        }
        let be = Arc::new(
            VulkanBackend::with_selection(DeviceSelection::Index(*idx))
                .unwrap_or_else(|e| panic!("open adapter {name:?}: {e:?}")),
        );
        let loc = fuel_ir::DeviceLocation::Vulkan { gpu_id: be.gpu_id };
        let mut bad = Vec::new();
        for (vname, data, shape, dim, corr, want) in vectors() {
            // Build on the CPU (a Vulkan device cannot hold a host-built const), then place every
            // COMPUTED node of the var recipe on this adapter, as the overlap benchmark does.
            let t = Tensor::from_f32(
                data.clone(),
                Shape::from_dims(&shape),
                &fuel_tensor::Device::cpu(),
            )
            .unwrap();
            let root = t.var(dim, corr).unwrap();
            {
                let gt = root.graph_tensor();
                let rid = gt.id();
                let graph = gt.graph();
                let ids = fuel_graph::topo_order(&graph.read().expect("graph lock"), rid);
                let mut g = graph.write().expect("graph lock");
                for id in ids {
                    if !g.node(id).inputs.is_empty() {
                        g.set_placement(id, loc);
                    }
                }
            }
            let got = root.realize_f32_vulkan(&be);
            let ok = got.iter().all(|g| {
                if want.is_nan() {
                    g.is_nan()
                } else {
                    *g == want
                }
            });
            println!(
                "VARVK {name} | {vname}: got {got:?} want {want} {}",
                if ok { "OK" } else { "DIVERGES" }
            );
            if !ok {
                bad.push(format!("{vname}: got {got:?} want {want}"));
            }
        }
        assert!(bad.is_empty(), "{name}: {bad:?}");
        ran += 1;
    }
    assert!(ran > 0, "no non-CPU Vulkan adapter found (GAP-157)");
}
