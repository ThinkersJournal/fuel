// SPDX-License-Identifier: MIT OR Apache-2.0
//! Captured-decode (`PipelinedExecutor::capture_decode`) replay tests on a live CUDA device, moved here from
//! `fuel-dispatch/src/pipelined.rs`'s unit tests (GAP-416). Since ROADMAP row 137 step 6.2 the CUDA kernels reach the
//! global binding table only through `fuel_cuda_dispatch::install_global()`, which `fuel-dispatch` (below it) cannot
//! call, so these failed on the first real run of the device job with "no backend supports <op>; available backends:
//! [Cpu]". Each test now installs the CUDA table first.
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch --features cuda
//! --test pipelined_capture_live -- --include-ignored --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_dispatch::pipelined::{CapturedDecodeSession, PipelinedExecutor, StorageCache};
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, DeviceLocation, SymEnv, probe::BackendId};
use fuel_memory::Storage;

/// CapturedRun executor build-out (Increment 1): capture a 2-op binary
/// chain `z = (a + b) * c` THROUGH the pipelined executor's
/// persistent-output mode, then prove the three capture guarantees on a
/// live GPU:
///   1. after capture, the fixed output buffer holds the warm-run result
///      (capture RECORDS launches, it does not execute them);
///   2. `replay()` recomputes bit-exactly into the SAME fixed buffer;
///   3. replay reads the FIXED input addresses — overwriting an input
///      buffer in place and replaying yields the new result, matching an
///      independent uncaptured `realize`. This is the whole mechanism:
///      the write-into binary wrapper + the executor's reuse mode +
///      `capture_run` compose into a replayable decode-step graph.
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_binary_chain_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    // z = (a + b) * c, shape [4]. `y = a + b` is a persistent
    // INTERMEDIATE; `z` is the persistent target — both fixed-address.
    let build = || -> (Arc<RwLock<Graph>>, NodeId, NodeId, NodeId, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a, b, c, z) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let c = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let y = g.push(Node {
                op: Op::Add,
                inputs: vec![a, b],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            let z = g.push(Node {
                op: Op::Mul,
                inputs: vec![y, c],
                shape: Shape::from_dims(&[4]),
                dtype: DType::F32,
            });
            g.set_target_backend(y, BackendId::Cuda);
            g.set_target_backend(z, BackendId::Cuda);
            (a, b, c, z)
        };
        (graph, a, b, c, z)
    };

    let up = |v: &[f32]| -> Arc<RwLock<Storage>> {
        let bytes: &[u8] = bytemuck::cast_slice(v);
        let cb = CudaStorageBytes::from_cpu_bytes(&dev, bytes).expect("h2d");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(cb),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        let host = c.to_cpu_bytes().expect("d2h");
        bytemuck::cast_slice(&host).to_vec()
    };

    let a_data = [1.0_f32, 2.0, 3.0, 4.0];
    let b_data = [10.0_f32, 20.0, 30.0, 40.0];
    let c_data = [2.0_f32, 2.0, 2.0, 2.0];
    let expect0 = [22.0_f32, 44.0, 66.0, 88.0]; // (a+b)*c

    // Fixed input Arcs (shared into capture; overwritten in place later).
    let a_arc = up(&a_data);
    let b_arc = up(&b_data);
    let c_arc = up(&c_data);

    let (graph, a, b, c, z) = build();
    let mut inputs = StorageCache::new();
    inputs.insert(a, a_arc.clone());
    inputs.insert(b, b_arc.clone());
    inputs.insert(c, c_arc.clone());

    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), z, inputs, SymEnv::default())
            .expect("capture_decode");

    // (1) warm-run result sits in the captured output buffer.
    assert_eq!(
        read(&captured.output),
        &expect0,
        "warm-run result after capture"
    );

    // (2) replay with the same inputs recomputes bit-exactly.
    captured.run.replay(&dev).expect("replay");
    dev.synchronize().expect("sync");
    assert_eq!(read(&captured.output), &expect0, "replay reproduces result");

    // (3) overwrite input `a` IN PLACE (same device address), replay ->
    //     the new result flows from the FIXED input address. Compare to
    //     an independent uncaptured realize with the same new input.
    let a2 = [100.0_f32, 200.0, 300.0, 400.0];
    {
        let g = a_arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("a not cuda")
        };
        c.write_from_host(bytemuck::cast_slice(&a2))
            .expect("in-place H2D");
    }
    captured.run.replay(&dev).expect("replay 2");
    dev.synchronize().expect("sync 2");
    let got = read(&captured.output);

    let (graph_ref, ar, br, cr, zr) = build();
    let mut inputs_ref = StorageCache::new();
    inputs_ref.insert(ar, up(&a2));
    inputs_ref.insert(br, up(&b_data));
    inputs_ref.insert(cr, up(&c_data));
    let (ref_arc, _) = PipelinedExecutor::realize(graph_ref, zr, inputs_ref).expect("ref realize");
    let expect1 = read(&ref_arc); // (a2+b)*c = [220,440,660,880]
    assert_eq!(
        got, expect1,
        "replay after in-place input update matches uncaptured realize"
    );
    assert_eq!(expect1, vec![220.0, 440.0, 660.0, 880.0]);
}

/// CapturedRun executor build-out (Increment 3): capture a DECODE-SHAPED
/// graph — the full dense op-set composing in one capture — and prove it
/// replays bit-exactly against an independent uncaptured realize.
///
/// Graph (an FFN block + norms, batch=seq=1 like real decode):
///   n1 = rms_norm(x);  u = n1 · w_up;  s = silu(u);
/// cuBLAS `gemm_dense` matmul IS graph-capturable (regression for the
/// dangling-input bug that once masqueraded as "cuBLAS not replay-safe").
/// Captures `y = a·w` where the caller does NOT retain `w`, FORCES the
/// freed-`w` slot to be reused by garbage buffers, then replays 50× — all
/// must equal the uncaptured realize. Passes iff `capture_decode` retains
/// its input buffers (`CapturedDecode::retained_inputs`).
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_matmul_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;
    let Ok(dev) = CudaDevice::new(0) else { return };
    const H: usize = 4;
    const F: usize = 8;
    let a_data: Vec<f32> = (0..H).map(|i| 0.5 + i as f32).collect();
    let w_data: Vec<f32> = (0..H * F).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let build = || -> (Arc<RwLock<Graph>>, NodeId, NodeId, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a, w, y) = {
            let mut g = graph.write().unwrap();
            let a = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[1, H]),
                dtype: DType::F32,
            });
            let w = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[H, F]),
                dtype: DType::F32,
            });
            let y = g.push(Node {
                op: Op::MatMul,
                inputs: vec![a, w],
                shape: Shape::from_dims(&[1, F]),
                dtype: DType::F32,
            });
            g.set_target_backend(y, BackendId::Cuda);
            (a, w, y)
        };
        (graph, a, w, y)
    };
    let up = |v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(CudaStorageBytes::from_cpu_bytes(&dev, &b).unwrap()),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        c.to_cpu_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    // Reference (uncaptured).
    let (gr, ar, wr, yr) = build();
    let mut insr = StorageCache::new();
    insr.insert(ar, up(&a_data));
    insr.insert(wr, up(&w_data));
    let (refarc, _) = PipelinedExecutor::realize(gr, yr, insr).unwrap();
    let expect = read(&refarc);
    // Capture (a + w NOT retained by caller).
    let (graph, a, w, y) = build();
    let mut inputs = StorageCache::new();
    inputs.insert(a, up(&a_data));
    inputs.insert(w, up(&w_data));
    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), y, inputs, SymEnv::default())
            .expect("capture");
    assert_eq!(read(&captured.output), expect, "warm result after capture");
    // FORCE freed-memory reuse: same-size (H*F) garbage buffers. If `w`
    // were dangling, replay would read 999.0s.
    let mut squatters = Vec::new();
    for _ in 0..8 {
        let g: Vec<u8> = (0..H * F).flat_map(|_| 999.0f32.to_le_bytes()).collect();
        squatters.push(CudaStorageBytes::from_cpu_bytes(&dev, &g).unwrap());
    }
    dev.synchronize().unwrap();
    let mut bad = 0usize;
    for _ in 0..50 {
        captured.run.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        if read(&captured.output) != expect {
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "cuBLAS matmul capture replays bit-exact ({bad}/50 wrong)"
    );
}

/// `Op::MulScalar` / `OpKind::Affine` (the `1/√d` attention scale) is
/// capture-safe (write-into variant + forced-reuse regression).
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_affine_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;
    let Ok(dev) = CudaDevice::new(0) else { return };
    const N: usize = 6;
    let x_data: Vec<f32> = (0..N).map(|i| 1.0 + i as f32).collect();
    let expect: Vec<f32> = x_data.iter().map(|v| v * 0.5).collect();
    let graph = Arc::new(RwLock::new(Graph::new()));
    let (x, y) = {
        let mut g = graph.write().unwrap();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[1, N]),
            dtype: DType::F32,
        });
        let y = g.push(Node {
            op: Op::MulScalar(0.5),
            inputs: vec![x],
            shape: Shape::from_dims(&[1, N]),
            dtype: DType::F32,
        });
        g.set_target_backend(y, BackendId::Cuda);
        (x, y)
    };
    let up = |v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(CudaStorageBytes::from_cpu_bytes(&dev, &b).unwrap()),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        c.to_cpu_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let mut inputs = StorageCache::new();
    inputs.insert(x, up(&x_data));
    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), y, inputs, SymEnv::default())
            .expect("capture");
    assert_eq!(read(&captured.output), expect, "warm result after capture");
    let mut sq = Vec::new();
    for _ in 0..8 {
        let g: Vec<u8> = (0..N).flat_map(|_| 999.0f32.to_le_bytes()).collect();
        sq.push(CudaStorageBytes::from_cpu_bytes(&dev, &g).unwrap());
    }
    dev.synchronize().unwrap();
    let mut bad = 0usize;
    for _ in 0..50 {
        captured.run.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        if read(&captured.output) != expect {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "affine capture replays bit-exact ({bad}/50 wrong)");
}

/// `index_select` (embedding lookup) IS graph-capturable (regression for
/// the dangling-input bug that once looked like "element 0 dropped on
/// replay"). Captures `emb = index_select(wte, tok)` with `wte` NOT
/// retained by the caller, FORCES the freed-`wte` slot to be reused by
/// garbage buffers, then replays 50× — all must be `[1,2,3,4]`. Passes iff
/// `capture_decode` retains its input buffers.
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_index_select_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;
    let Ok(dev) = CudaDevice::new(0) else { return };
    const V: usize = 3;
    const H: usize = 4;
    let wte_data: Vec<f32> = (0..V)
        .flat_map(|v| (0..H).map(move |c| 1.0 + 10.0 * v as f32 + c as f32))
        .collect();
    let graph = Arc::new(RwLock::new(Graph::new()));
    let (tok, wte, emb) = {
        let mut g = graph.write().unwrap();
        let tok = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[1]),
            dtype: DType::U32,
        });
        let wte = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[V, H]),
            dtype: DType::F32,
        });
        let emb = g.push(Node {
            op: Op::IndexSelect { dim: 0 },
            inputs: vec![wte, tok],
            shape: Shape::from_dims(&[1, H]),
            dtype: DType::F32,
        });
        g.set_target_backend(emb, BackendId::Cuda);
        (tok, wte, emb)
    };
    let up_f32 = |v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(CudaStorageBytes::from_cpu_bytes(&dev, &b).unwrap()),
            DType::F32,
        )))
    };
    let up_u32 = |v: &[u32]| {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(CudaStorageBytes::from_cpu_bytes(&dev, &b).unwrap()),
            DType::U32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        c.to_cpu_bytes()
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let tok_arc = up_u32(&[0]);
    let mut inputs = StorageCache::new();
    inputs.insert(tok, tok_arc.clone());
    inputs.insert(wte, up_f32(&wte_data));
    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), emb, inputs, SymEnv::default())
            .expect("capture");
    assert_eq!(
        read(&captured.output),
        vec![1.0, 2.0, 3.0, 4.0],
        "warm result after capture"
    );
    // FORCE freed-memory reuse: same-size (V*H) garbage buffers reclaim
    // `wte`'s slot if it were dangling — replay would then read 999.0s.
    let mut squatters = Vec::new();
    for _ in 0..8 {
        let g: Vec<u8> = (0..V * H).flat_map(|_| 999.0f32.to_le_bytes()).collect();
        squatters.push(CudaStorageBytes::from_cpu_bytes(&dev, &g).unwrap());
    }
    dev.synchronize().unwrap();
    let mut bad = 0usize;
    for _ in 0..50 {
        captured.run.replay(&dev).unwrap();
        dev.synchronize().unwrap();
        if read(&captured.output) != vec![1.0, 2.0, 3.0, 4.0] {
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "index_select capture replays bit-exact ({bad}/50 wrong)"
    );
}

///   n1 = rms_norm(x);  s = silu(n1);  p = softmax(s);
///   r  = p + x;  n2 = rms_norm(r);  y = n2 + x
/// Exercises rms_norm (×2 — proves the per-device RMS workspace is reused
/// across sequential norms inside ONE captured graph), silu + softmax
/// (write-into), and residual adds through the persistent-output capture
/// path. Verified across two input value-sets (robust, not luck).
///
/// Deliberately EXCLUDES matmul (cuBLAS not graph-replay-safe) and
/// index_select (its host-side shape/stride arrays dangle under capture —
/// proven to mis-replay). Both are surfaced `capture_decode` gaps, not
/// silent mis-replays. See `op_kind_is_capture_writeinto`.
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_norm_silu_softmax_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_graph::registry::{FusedOpParams, FusedOps};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };
    const H: usize = 6;
    let eps = 1e-5f64;

    let build = || -> (Arc<RwLock<Graph>>, NodeId, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x, y) = {
            let mut g = graph.write().unwrap();
            let mk = |g: &mut Graph, op: Op, ins: Vec<NodeId>, dims: &[usize]| {
                g.push(Node {
                    op,
                    inputs: ins,
                    shape: Shape::from_dims(dims),
                    dtype: DType::F32,
                })
            };
            let rms = || {
                Op::Fused(
                    FusedOps::RMS_NORM_LAST_DIM,
                    FusedOpParams::RmsNormLastDim { eps },
                )
            };
            let sm = || Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim);
            let x = mk(&mut g, Op::Const, vec![], &[1, H]);
            let n1 = mk(&mut g, rms(), vec![x], &[1, H]);
            let s = mk(&mut g, Op::Silu, vec![n1], &[1, H]);
            let p = mk(&mut g, sm(), vec![s], &[1, H]);
            let r = mk(&mut g, Op::Add, vec![p, x], &[1, H]);
            let n2 = mk(&mut g, rms(), vec![r], &[1, H]);
            let y = mk(&mut g, Op::Add, vec![n2, x], &[1, H]);
            for id in [n1, s, p, r, n2, y] {
                g.set_target_backend(id, BackendId::Cuda);
            }
            (x, y)
        };
        (graph, x, y)
    };

    let up = |v: &[f32]| -> Arc<RwLock<Storage>> {
        let cb = CudaStorageBytes::from_cpu_bytes(&dev, bytemuck::cast_slice(v)).expect("h2d");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(cb),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        bytemuck::cast_slice(&c.to_cpu_bytes().expect("d2h")).to_vec()
    };

    let x_data: Vec<f32> = (0..H).map(|c| 0.1 * (c as f32 + 1.0)).collect();
    let x_arc = up(&x_data);
    let (graph, x, y) = build();
    let mut inputs = StorageCache::new();
    inputs.insert(x, x_arc.clone());

    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), y, inputs, SymEnv::default())
            .expect("capture_decode norm/silu/softmax");

    let realize_ref = |xv: &[f32]| -> Vec<f32> {
        let (g, xr, yr) = build();
        let mut ins = StorageCache::new();
        ins.insert(xr, up(xv));
        let (arc, _) = PipelinedExecutor::realize(g, yr, ins).expect("ref realize");
        read(&arc)
    };
    let expect0 = realize_ref(&x_data);

    // (1) after capture the output holds the warm result.
    assert_eq!(read(&captured.output), expect0, "warm result after capture");
    // (2) replay reproduces bit-exactly (incl. RMS workspace reuse on replay).
    captured.run.replay(&dev).expect("replay");
    dev.synchronize().expect("sync");
    assert_eq!(read(&captured.output), expect0, "replay reproduces result");

    // (3) overwrite x in place, replay -> matches uncaptured realize (a
    //     DIFFERENT value-set — so the chain is robust, not input-lucky).
    let x2: Vec<f32> = (0..H).map(|c| 0.3 * (c as f32 + 1.0)).collect();
    {
        let g = x_arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("x not cuda")
        };
        c.write_from_host(bytemuck::cast_slice(&x2))
            .expect("in-place H2D");
    }
    captured.run.replay(&dev).expect("replay 2");
    dev.synchronize().expect("sync 2");
    assert_eq!(
        read(&captured.output),
        realize_ref(&x2),
        "replay after x update matches realize"
    );
}

/// CapturedRun executor build-out (Increment 4b-γ): make `ContiguizeOf`
/// (a non-contiguous reshape after `Permute` — the real decode graph's
/// attention-merge reshape shape) AND a 2-input `Op::Concat` (the shape
/// RoPE's CUDA decomposition emits for its rotate-half) BOTH
/// capture-safe in ONE graph, so the crux 4b-γ exists for — allocation
/// of an INTERMEDIATE, non-final buffer inside a capture scope — is
/// actually exercised, not just each gap in isolation.
///
/// Graph: `x [2,3,4] --Permute[1,0,2]--> [3,2,4]` (non-contiguous)
/// `--Reshape[3,8]--> r` (`ContiguizeOf`, MUST allocate: its input is
/// non-contiguous) `--Concat(dim=1, with b [3,8])--> y [3,16]` (target).
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_contiguize_then_concat_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let build = || -> (Arc<RwLock<Graph>>, NodeId, NodeId, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x, b, y) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2, 3, 4]),
                dtype: DType::F32,
            });
            let p = g.push(Node {
                op: Op::Permute(vec![1, 0, 2]),
                inputs: vec![x],
                shape: Shape::from_dims(&[3, 2, 4]),
                dtype: DType::F32,
            });
            let r = g.push(Node {
                op: Op::Reshape(Shape::from_dims(&[3, 8])),
                inputs: vec![p],
                shape: Shape::from_dims(&[3, 8]),
                dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3, 8]),
                dtype: DType::F32,
            });
            let y = g.push(Node {
                op: Op::Concat { dim: 1 },
                inputs: vec![r, b],
                shape: Shape::from_dims(&[3, 16]),
                dtype: DType::F32,
            });
            g.set_target_backend(p, BackendId::Cuda);
            g.set_target_backend(r, BackendId::Cuda);
            g.set_target_backend(y, BackendId::Cuda);
            (x, b, y)
        };
        (graph, x, b, y)
    };

    let up = |v: &[f32]| -> Arc<RwLock<Storage>> {
        let cb = CudaStorageBytes::from_cpu_bytes(&dev, bytemuck::cast_slice(v)).expect("h2d");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(cb),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        bytemuck::cast_slice(&c.to_cpu_bytes().expect("d2h")).to_vec()
    };

    let x_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..24).map(|i| 100.0 + i as f32).collect();
    let x_arc = up(&x_data);
    let (graph, x, b, y) = build();
    let mut inputs = StorageCache::new();
    inputs.insert(x, x_arc.clone());
    // `b` deliberately NOT retained by a local variable — forces
    // reliance on `capture_decode`'s own `retained_inputs`, mirroring
    // the matmul/index_select regression precedent.
    inputs.insert(b, up(&b_data));

    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), y, inputs, SymEnv::default())
            .expect("capture_decode contiguize+concat");

    let realize_ref = |xv: &[f32], bv: &[f32]| -> Vec<f32> {
        let (g, xr, br, yr) = build();
        let mut ins = StorageCache::new();
        ins.insert(xr, up(xv));
        ins.insert(br, up(bv));
        let (arc, _) = PipelinedExecutor::realize(g, yr, ins).expect("ref realize");
        read(&arc)
    };
    let expect0 = realize_ref(&x_data, &b_data);

    // (1) warm-run result sits in the captured output buffer (capture
    //     RECORDS launches, it does not execute them at capture time).
    assert_eq!(read(&captured.output), expect0, "warm result after capture");

    // (2) replay TWICE with unchanged inputs — proves the fixed-address
    //     reuse (both the ContiguizeOf intermediate and the Concat
    //     output) holds across multiple replay() calls, not just one.
    for i in 0..2 {
        captured
            .run
            .replay(&dev)
            .unwrap_or_else(|e| panic!("replay {i}: {e:?}"));
        dev.synchronize().expect("sync");
        assert_eq!(
            read(&captured.output),
            expect0,
            "replay {i} reproduces result"
        );
    }

    // (3) overwrite x in place (a DIFFERENT value-set), replay again ->
    //     the new result flows from the FIXED input address, through
    //     the fixed ContiguizeOf intermediate, into the fixed Concat
    //     output — matches an independent uncaptured realize.
    let x2: Vec<f32> = (0..24).map(|i| 1000.0 + i as f32).collect();
    {
        let g = x_arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("x not cuda")
        };
        c.write_from_host(bytemuck::cast_slice(&x2))
            .expect("in-place H2D");
    }
    captured.run.replay(&dev).expect("replay 3");
    dev.synchronize().expect("sync 3");
    assert_eq!(
        read(&captured.output),
        realize_ref(&x2, &b_data),
        "replay after x update matches independent uncaptured realize"
    );
}

/// CapturedRun executor build-out (Increment 4b-ζ): a same-device
/// CUDA→CUDA `Op::Copy` — the shape `insert_safety_copies` emits for its
/// aliasing-hazard snapshots in the real Llama decode graph — is
/// capture-safe via `copy_from_cuda_wrapper`'s write-into
/// `BackendStorage::Cuda` branch (`CudaStorageBytes::copy_from_device`).
///
/// Graph: `x [2,4]`, `b [2,4]` --Copy{target: Cuda{0}}--> `c` (same-device
/// snapshot of `x`) --Add(c, b)--> `y`. Capture, replay twice unchanged,
/// then update `x` in place and replay again — every step must match an
/// independent uncaptured `realize`.
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_same_device_copy_replays_bit_exact_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let build = || -> (Arc<RwLock<Graph>>, NodeId, NodeId, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x, b, y) = {
            let mut g = graph.write().unwrap();
            let x = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2, 4]),
                dtype: DType::F32,
            });
            let b = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[2, 4]),
                dtype: DType::F32,
            });
            // Same-device CUDA->CUDA copy — the write-into-safe case.
            let c = g.push(Node {
                op: Op::Copy {
                    target: DeviceLocation::Cuda { gpu_id: 0 },
                },
                inputs: vec![x],
                shape: Shape::from_dims(&[2, 4]),
                dtype: DType::F32,
            });
            let y = g.push(Node {
                op: Op::Add,
                inputs: vec![c, b],
                shape: Shape::from_dims(&[2, 4]),
                dtype: DType::F32,
            });
            // Copy's target_backend = SOURCE backend (x is Cuda-resident).
            g.set_target_backend(c, BackendId::Cuda);
            g.set_target_backend(y, BackendId::Cuda);
            (x, b, y)
        };
        (graph, x, b, y)
    };

    let up = |v: &[f32]| -> Arc<RwLock<Storage>> {
        let cb = CudaStorageBytes::from_cpu_bytes(&dev, bytemuck::cast_slice(v)).expect("h2d");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(cb),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        bytemuck::cast_slice(&c.to_cpu_bytes().expect("d2h")).to_vec()
    };

    let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..8).map(|i| 100.0 + i as f32).collect();
    let x_arc = up(&x_data);
    let (graph, x, b, y) = build();
    let mut inputs = StorageCache::new();
    inputs.insert(x, x_arc.clone());
    // `b` deliberately NOT retained by a local variable — forces
    // reliance on `capture_decode`'s own `retained_inputs`.
    inputs.insert(b, up(&b_data));

    let captured =
        PipelinedExecutor::capture_decode(Arc::clone(&graph), y, inputs, SymEnv::default())
            .expect("capture_decode same-device copy");

    let realize_ref = |xv: &[f32], bv: &[f32]| -> Vec<f32> {
        let (g, xr, br, yr) = build();
        let mut ins = StorageCache::new();
        ins.insert(xr, up(xv));
        ins.insert(br, up(bv));
        let (arc, _) = PipelinedExecutor::realize(g, yr, ins).expect("ref realize");
        read(&arc)
    };
    let expect0 = realize_ref(&x_data, &b_data);

    // (1) warm-run result sits in the captured output buffer.
    assert_eq!(read(&captured.output), expect0, "warm result after capture");

    // (2) replay TWICE with unchanged inputs — the Copy's write-into
    //     destination buffer must stay bit-exact across replays.
    for i in 0..2 {
        captured
            .run
            .replay(&dev)
            .unwrap_or_else(|e| panic!("replay {i}: {e:?}"));
        dev.synchronize().expect("sync");
        assert_eq!(
            read(&captured.output),
            expect0,
            "replay {i} reproduces result"
        );
    }

    // (3) overwrite x in place (a DIFFERENT value-set), replay again ->
    //     the Copy re-runs on the fixed input address into its fixed
    //     destination buffer -> matches an independent uncaptured realize.
    let x2: Vec<f32> = (0..8).map(|i| 1000.0 + i as f32).collect();
    {
        let g = x_arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("x not cuda")
        };
        c.write_from_host(bytemuck::cast_slice(&x2))
            .expect("in-place H2D");
    }
    captured.run.replay(&dev).expect("replay 3");
    dev.synchronize().expect("sync 3");
    assert_eq!(
        read(&captured.output),
        realize_ref(&x2, &b_data),
        "replay after x update matches independent uncaptured realize"
    );
}

/// CapturedRun executor build-out (Increment 4b-ζ): a genuinely
/// cross-device `Op::Copy` (D2H — `target: DeviceLocation::Cpu` from a
/// CUDA-resident source) must still be REJECTED by `capture_decode`'s
/// validation loop — a host round-trip can never be part of a CUDA
/// graph capture. Fast check: no replay, just confirm the typed `Err`.
#[test]
#[ignore = "requires a live CUDA device"]
fn capture_decode_rejects_cross_device_copy_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };

    let graph = Arc::new(RwLock::new(Graph::new()));
    let (x, y) = {
        let mut g = graph.write().unwrap();
        let x = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(&[2, 4]),
            dtype: DType::F32,
        });
        // Genuinely cross-device: D2H.
        let y = g.push(Node {
            op: Op::Copy {
                target: DeviceLocation::Cpu,
            },
            inputs: vec![x],
            shape: Shape::from_dims(&[2, 4]),
            dtype: DType::F32,
        });
        g.set_target_backend(y, BackendId::Cuda); // source backend = Cuda
        (x, y)
    };

    let x_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let cb = CudaStorageBytes::from_cpu_bytes(&dev, bytemuck::cast_slice(&x_data)).expect("h2d");
    let x_arc = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Cuda(cb),
        DType::F32,
    )));
    let mut inputs = StorageCache::new();
    inputs.insert(x, x_arc);

    let result = PipelinedExecutor::capture_decode(graph, y, inputs, SymEnv::default());
    let msg = match result {
        Ok(_) => panic!("D2H Op::Copy must be rejected by capture_decode"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("cross-device"),
        "error should describe the cross-device rejection: {msg}"
    );
}

/// CapturedDecodeSession (Increment 4a): drive the warm→capture→replay
/// orchestration over MANY tokens on a synthetic decode-shaped session —
/// the real per-token decode pattern. Capture once, then per token
/// H2D-update the fixed input buffer IN PLACE + replay, asserting each
/// token's output is bit-exact against an independent uncaptured realize.
/// A kernel-agnostic proof of the state machine + fixed-buffer mechanics
/// (matmul + index_select excluded pending baracuda capture-safe kernels).
#[test]
#[ignore = "requires a live CUDA device"]
fn captured_decode_session_replays_many_tokens_cuda() {
    fuel_cuda_dispatch::install_global();
    use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
    use fuel_graph::registry::{FusedOpParams, FusedOps};
    use fuel_ir::Shape;
    use fuel_memory::BackendStorage;

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };
    const H: usize = 6;
    let eps = 1e-5f64;

    let build = || -> (Arc<RwLock<Graph>>, NodeId, NodeId) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x, y) = {
            let mut g = graph.write().unwrap();
            let mk = |g: &mut Graph, op: Op, ins: Vec<NodeId>, dims: &[usize]| {
                g.push(Node {
                    op,
                    inputs: ins,
                    shape: Shape::from_dims(dims),
                    dtype: DType::F32,
                })
            };
            let rms = || {
                Op::Fused(
                    FusedOps::RMS_NORM_LAST_DIM,
                    FusedOpParams::RmsNormLastDim { eps },
                )
            };
            let sm = || Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim);
            let x = mk(&mut g, Op::Const, vec![], &[1, H]);
            let n1 = mk(&mut g, rms(), vec![x], &[1, H]);
            let s = mk(&mut g, Op::Silu, vec![n1], &[1, H]);
            let p = mk(&mut g, sm(), vec![s], &[1, H]);
            let r = mk(&mut g, Op::Add, vec![p, x], &[1, H]);
            let n2 = mk(&mut g, rms(), vec![r], &[1, H]);
            let y = mk(&mut g, Op::Add, vec![n2, x], &[1, H]);
            for id in [n1, s, p, r, n2, y] {
                g.set_target_backend(id, BackendId::Cuda);
            }
            (x, y)
        };
        (graph, x, y)
    };
    let up = |v: &[f32]| -> Arc<RwLock<Storage>> {
        let cb = CudaStorageBytes::from_cpu_bytes(&dev, bytemuck::cast_slice(v)).expect("h2d");
        Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(cb),
            DType::F32,
        )))
    };
    let read = |arc: &Arc<RwLock<Storage>>| -> Vec<f32> {
        let g = arc.read().unwrap();
        let BackendStorage::Cuda(c) = &g.inner else {
            panic!("not cuda")
        };
        bytemuck::cast_slice(&c.to_cpu_bytes().expect("d2h")).to_vec()
    };
    let realize_ref = |xv: &[f32]| -> Vec<f32> {
        let (g, xr, yr) = build();
        let mut ins = StorageCache::new();
        ins.insert(xr, up(xv));
        let (arc, _) = PipelinedExecutor::realize(g, yr, ins).expect("ref realize");
        read(&arc)
    };

    // 5 distinct "tokens" (distinct per-token input value-sets).
    let tokens: Vec<Vec<f32>> = (0..5)
        .map(|t| {
            (0..H)
                .map(|c| 0.1 * (t as f32 + 1.0) * (c as f32 + 1.0))
                .collect()
        })
        .collect();

    // Capture ONCE against token 0's fixed input buffer.
    let x_arc = up(&tokens[0]);
    let (graph, x, y) = build();
    let mut inputs = StorageCache::new();
    inputs.insert(x, x_arc.clone());
    let session =
        CapturedDecodeSession::capture(Arc::clone(&graph), y, inputs, &[x], SymEnv::default())
            .expect("capture session");

    // Warm result (token 0) already sits in the output buffer.
    assert_eq!(
        read(session.output()),
        realize_ref(&tokens[0]),
        "warm token 0"
    );

    // The decode loop: replay every token (incl. re-replaying token 0) via
    // in-place H2D update of the fixed input + one graph launch.
    for (t, tok) in tokens.iter().enumerate() {
        let out = session
            .replay_token(&[(x, bytemuck::cast_slice(tok))])
            .expect("replay_token");
        assert_eq!(
            read(&out),
            realize_ref(tok),
            "replayed token {t} matches uncaptured realize"
        );
    }
    // Distinct tokens must give distinct outputs (live recompute, not stale).
    assert_ne!(
        realize_ref(&tokens[0]),
        realize_ref(&tokens[1]),
        "tokens must differ"
    );
}
