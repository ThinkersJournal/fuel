// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-GPU coverage of the CUDA `Op::Iota` arm (GAP-380 / ROADMAP row 149 (a2)): an `Iota` stamped
//! for CUDA is generated on the host and uploaded, through the PRODUCTION `PipelinedExecutor`. (A CUDA
//! KERNEL reading it is covered by `fuel-model-llama`'s `bf16_paged_decode_matches_contiguous_on_cuda`:
//! `fuel-dispatch` cannot register CUDA kernels itself, they live in `fuel-cuda-dispatch`.) Before the arm existed the executor answered
//! `WorkItemKind::Iota: only CPU is wired today, got Cuda`.
//!
//! REQUIRES an NVIDIA GPU + CUDA Runtime SDK and says so by FAILING without one (GAP-157: an early return
//! reports `ok` having asserted nothing, and `#[ignore]` is capped by `suppression_ratchet`). The
//! `cuda` feature keeps it out of every CI build. Run it through the machine-wide lock:
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-dispatch --features cuda \
//!     --test cuda_iota_live -- --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::CudaDevice;
use fuel_dispatch::pipelined::{DeviceRegistry, PipelinedExecutor, StorageCache};
use fuel_graph::{Graph, Node, NodeId, Op, SharedGraph};
use fuel_ir::{DType, DeviceLocation, Shape, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};

fn require_device() -> CudaDevice {
    CudaDevice::new(0).expect(
        "this test REQUIRES a live CUDA device (GAP-157: it must fail, not skip, without one)",
    )
}

/// `Iota{len}` stamped for CUDA GPU 0. Returns `(graph, root)`.
fn build(len: usize) -> (SharedGraph, NodeId) {
    let graph: SharedGraph = Arc::new(RwLock::new(Graph::new()));
    let root = {
        let mut g = graph.write().unwrap();
        let iota = g.push(Node {
            op: Op::Iota { len },
            inputs: vec![],
            shape: Shape::from_dims(&[len]),
            dtype: DType::F32,
        });
        g.set_target_backend(iota, BackendId::Cuda);
        g.set_placement(iota, DeviceLocation::Cuda { gpu_id: 0 });
        iota
    };
    (graph, root)
}

fn realize_on_cuda(dev: &CudaDevice, len: usize) -> Vec<f32> {
    let (graph, root) = build(len);
    let (out, _) = PipelinedExecutor::realize_with_devices(
        graph,
        root,
        StorageCache::new(),
        DeviceRegistry::new().with_cuda(dev.clone()),
    )
    .expect("realize an Iota graph on CUDA");
    let guard = out.read().unwrap();
    let BackendStorage::Cuda(c) = &guard.inner else {
        panic!("expected the output resident on CUDA, not the host");
    };
    let host = c.to_cpu_bytes().expect("d2h");
    bytemuck::cast_slice::<u8, f32>(&host).to_vec()
}

#[test]
fn a_cuda_iota_is_the_position_ramp_and_stays_on_the_device() {
    let dev = require_device();
    assert_eq!(realize_on_cuda(&dev, 4), vec![0.0_f32, 1.0, 2.0, 3.0]);
    // A length that is not a power of two or a warp multiple: the upload is exact-sized.
    let v = realize_on_cuda(&dev, 37);
    assert_eq!(v.len(), 37);
    assert!(
        v.iter().enumerate().all(|(i, &x)| x == i as f32),
        "ramp 0..37: {v:?}"
    );
}
