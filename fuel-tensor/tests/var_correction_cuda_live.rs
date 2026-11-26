// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Tensor::var`'s divisor clamp (`max(0, n - correction)`, the torch rule; ruling 2026-11-25) run
//! through the CUDA backend (f32). `var` is a RECIPE over mean/sub/sqr/sum/mul_scalar, so CUDA
//! inherits the rule through its own kernels; this MEASURES it. The same 7 vectors as the CPU test
//! `phase_a3_keepdim_tests::var_divisor_is_clamped_at_zero_in_every_dtype` and the Vulkan test
//! `var_correction_vulkan_live`.
//!
//! **Limit of this measurement:** every COMPUTED node of the recipe is stamped to the CUDA device
//! and the graph is realized through `realize_f32_cuda`; there is no counter proving each kernel ran
//! on the GPU, so the claim is "the CUDA realize path of the placed graph gives these values".
//!
//! Gate: `require_gpu_run_lock`; a missing device is a FAILURE. Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-tensor --features cuda --test var_correction_cuda_live -- --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use fuel_cuda_backend::CudaDevice;
use fuel_ir::{DeviceLocation, Shape};
use fuel_tensor::lazy::Tensor;

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
fn var_divisor_clamp_on_cuda() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let loc = DeviceLocation::Cuda { gpu_id: 0 };
    let mut bad = Vec::new();
    for (vname, data, shape, dim, corr, want) in vectors() {
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
        let got = root.realize_f32_cuda(&dev);
        let ok = got.iter().all(|g| {
            if want.is_nan() {
                g.is_nan()
            } else {
                *g == want
            }
        });
        println!(
            "VARCUDA {vname}: got {got:?} want {want} {}",
            if ok { "OK" } else { "DIVERGES" }
        );
        if !ok {
            bad.push(format!("{vname}: got {got:?} want {want}"));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}
