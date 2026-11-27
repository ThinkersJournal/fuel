// SPDX-License-Identifier: MIT OR Apache-2.0
//! What a MASKED attention does on CUDA in f16 / bf16 / f32 when the fused kernel is unreachable (prefill, `sq > 1`,
//! or a window): the graph decomposes, and since fuel#716 the decomposition masks the scores with
//! `MaskedFill(-inf)` in the attention dtype, for which the CUDA binding census has f32/f64/i32 only
//! (baracuda E3 ranking, pm-kernels 2026-11-26). This MEASURES the outcome per dtype: `OK` (agrees with the CPU f32
//! reference within the dtype's tolerance), `DIVERGES` (a silent wrong result), or `ERR` (a panic / typed failure,
//! message printed). f32 is the positive control. Cases: a plain causal prefill (4x4) and the mixed fully-masked row.
//!
//! **Limit:** every COMPUTED node is stamped to the CUDA device and the graph realized through `realize_f32_cuda`;
//! there is no counter proving each kernel ran on the GPU, so "ran on CUDA" is the placement + realize path, and a
//! host fallback inside it would read as OK. The printed `ERR`/`OK` is the observable.
//!
//! Gate: `require_gpu_run_lock`; a missing device is a FAILURE. Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-tensor --features cuda --test flash_attn_half_masked_cuda_live -- --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use fuel_cuda_backend::CudaDevice;
use fuel_ir::{DType, DeviceLocation, Shape};
use fuel_tensor::lazy::Tensor;

struct Case {
    name: &'static str,
    sq: usize,
    sk: usize,
    wl: Option<usize>,
}

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.8)
        .collect()
}

fn attn(c: &Case, dt: DType, dev: &fuel_tensor::Device) -> Tensor {
    let d = 8;
    let cast = |t: Tensor| {
        if dt == DType::F32 {
            t
        } else {
            t.to_dtype(dt).unwrap()
        }
    };
    // k and v must live on q's graph (`const_f32_like`).
    let q32 =
        Tensor::from_f32(data(c.sq * d, 0.1), Shape::from_dims(&[1, 1, c.sq, d]), dev).unwrap();
    let k32 = q32
        .const_f32_like(data(c.sk * d, 0.7), Shape::from_dims(&[1, 1, c.sk, d]))
        .unwrap();
    let v32 = q32
        .const_f32_like(data(c.sk * d, 1.3), Shape::from_dims(&[1, 1, c.sk, d]))
        .unwrap();
    let (q, k, v) = (cast(q32), cast(k32), cast(v32));
    let o = q
        .flash_attn(&k, &v, None, 0.35, true, c.wl, None, None)
        .unwrap();
    if dt == DType::F32 {
        o
    } else {
        o.to_dtype(DType::F32).unwrap()
    }
}

fn stamp_cuda(
    graph: &std::sync::Arc<std::sync::RwLock<fuel_graph::Graph>>,
    root: fuel_graph::NodeId,
) {
    let loc = DeviceLocation::Cuda { gpu_id: 0 };
    let ids = fuel_graph::topo_order(&graph.read().expect("graph lock"), root);
    let mut g = graph.write().expect("graph lock");
    for id in ids {
        if !g.node(id).inputs.is_empty() {
            g.set_placement(id, loc);
        }
    }
}

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {
    let msg = p
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    msg.replace(char::from(10), " | ")
        .chars()
        .take(700)
        .collect()
}

fn verdict(got: &[f32], want: &[f32], tol: f32) -> String {
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| {
            if g.is_nan() || w.is_nan() {
                f32::INFINITY
            } else {
                (g - w).abs()
            }
        })
        .fold(0.0f32, f32::max);
    let kind = if got.len() == want.len() && worst <= tol {
        "OK"
    } else {
        "DIVERGES"
    };
    format!(
        "{kind} worst_abs_diff={worst:e} tol={tol:e} got[0..4]={:?}",
        &got[..got.len().min(4)]
    )
}

#[test]
fn half_masked_attention_on_cuda() {
    fuel_test_support::require_gpu_run_lock();
    let cuda = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let fc_device: fuel_tensor::Device = cuda.clone().into();
    let cpu = fuel_tensor::Device::cpu();
    let cases = [
        Case {
            name: "causal-prefill-4x4",
            sq: 4,
            sk: 4,
            wl: None,
        },
        Case {
            name: "fully-masked-row-mixed",
            sq: 3,
            sk: 2,
            wl: Some(0),
        },
    ];
    for c in &cases {
        let want = attn(c, DType::F32, &cpu).realize_f32();
        for (dt, tol) in [
            (DType::F32, 1e-4f32),
            (DType::BF16, 5e-2),
            (DType::F16, 2e-2),
        ] {
            // (a) the graph as built: realize on CUDA (the optimizer decides decompose vs fused)
            let root = attn(c, dt, &cpu);
            {
                let gt = root.graph_tensor();
                stamp_cuda(&gt.graph().clone(), gt.id());
            }
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                root.realize_f32_cuda(&cuda)
            }));
            match out {
                Err(p) => println!(
                    "MASKEDHALF {} {dt:?} [as-built]: ERR {}",
                    c.name,
                    panic_text(p)
                ),
                Ok(got) => println!(
                    "MASKEDHALF {} {dt:?} [as-built]: {}",
                    c.name,
                    verdict(&got, &want, tol)
                ),
            }
            // (b) the graph LOWERED to primitives first (the decomposition), then placed on CUDA
            let root = attn(c, dt, &cpu);
            let gt = root.graph_tensor();
            let graph = gt.graph().clone();
            let roots = fuel_graph::opt::RuleRegistry::lowering_only()
                .optimize_to_fixpoint(&graph, &[gt.id()]);
            stamp_cuda(&graph, roots[0]);
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fuel_tensor::pipelined_bridge::realize_one_as::<f32>(&graph, roots[0], &fc_device)
            }));
            match out {
                Err(p) => println!(
                    "MASKEDHALF {} {dt:?} [lowered]: PANIC {}",
                    c.name,
                    panic_text(p)
                ),
                Ok(Err(e)) => println!(
                    "MASKEDHALF {} {dt:?} [lowered]: ERR {}",
                    c.name,
                    format!("{e:?}").chars().take(700).collect::<String>()
                ),
                Ok(Ok(got)) => println!(
                    "MASKEDHALF {} {dt:?} [lowered]: {}",
                    c.name,
                    verdict(&got, &want, tol)
                ),
            }
        }
    }
}
