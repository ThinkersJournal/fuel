// SPDX-License-Identifier: MIT OR Apache-2.0
//! JIT probe-reference and PagedAttn placement-report tests on a live CUDA device, moved here from
//! `fuel-dispatch`'s unit tests (GAP-418): since ROADMAP row 137 step 6.2 the CUDA kernels reach the global
//! binding table only through `fuel_cuda_dispatch::install_global()`, which `fuel-dispatch` cannot call, so they
//! failed on the first real run of the device job ("no backend supports <op>; available backends: [Cpu]").
//! They reach the private JIT probe functions through the hidden, semver-exempt
//! `fuel_dispatch::jit_test_exports` module (feature `dev-tools`).
//!
//! Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch --features jit
//! --test jit_probe_live -- --include-ignored --test-threads=1`.

#![cfg(feature = "jit")]

use fuel_dispatch::jit_test_exports::{
    HostTensor, fill_deterministic, op_to_op_kind, reference_from_registered_recipe,
    reference_output,
};
use fuel_dispatch::optimize::optimize_graph;
use fuel_dispatch::plan::PlanOptions;
use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::{DType, DeviceLocation, Shape, dispatch::OpKind, probe::BackendId};

/// Build a contiguous F32 `HostTensor` of shape `[vals.len()]`.
fn ht_f32(vals: &[f32]) -> HostTensor {
    HostTensor {
        dtype: DType::F32,
        shape: vec![vals.len()],
        bytes: bytemuck::cast_slice(vals).to_vec(),
    }
}

/// Reinterpret a byte buffer as `f32`s (little-endian, native).
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
}

/// Deterministic F32 rope probe: `x` `[1, seq, head_dim]` plus cos/sin `[seq, head_dim]`.
fn rope_probe(
    seq: usize,
    head_dim: usize,
    base: f64,
) -> (HostTensor, HostTensor, HostTensor, Vec<usize>) {
    use fuel_graph::build_rope_tables;
    let x_shape = vec![1usize, seq, head_dim];
    let x_vals = fill_deterministic(seq * head_dim, 0x5EED_D07);
    let (cos, sin) = build_rope_tables(base, 0, seq, head_dim);
    let x = HostTensor {
        dtype: DType::F32,
        shape: x_shape.clone(),
        bytes: bytemuck::cast_slice(&x_vals).to_vec(),
    };
    let cos_t = HostTensor {
        dtype: DType::F32,
        shape: vec![seq, head_dim],
        bytes: bytemuck::cast_slice(&cos).to_vec(),
    };
    let sin_t = HostTensor {
        dtype: DType::F32,
        shape: vec![seq, head_dim],
        bytes: bytemuck::cast_slice(&sin).to_vec(),
    };
    (x, cos_t, sin_t, x_shape)
}

/// Host-side rotate-half rope: `out = x*cos + rotate_half(x)*sin`.
fn expected_rotate_half(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = vec![0.0f32; seq * head_dim];
    for s in 0..seq {
        let row = s * head_dim;
        for d in 0..head_dim {
            let rot = if d < half {
                -x[row + d + half]
            } else {
                x[row + d - half]
            };
            out[row + d] = x[row + d] * cos[row + d] + rot * sin[row + d];
        }
    }
    out
}

/// Live-GPU (Spec-B Task-4 acceptance test): `reference_from_registered_recipe`
/// builds `Op::Fused(FusedOps::ROPE, FusedOpParams::Rope)` on rope-shaped F32
/// probes, lowers it to Fuel's registered primitive base map, realizes it on
/// CUDA, and returns bytes equal (F32 tolerance) to a host-computed rotate-half
/// rope. `#[ignore]`'d (needs a live CUDA device).
#[test]
#[ignore = "requires a live CUDA device"]
fn reference_from_registered_recipe_realizes_rotate_half_rope() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::registry::{FusedOpParams, FusedOps};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };
    let (seq, head_dim, base) = (4usize, 8usize, 10000.0f64);
    let (x, cos_t, sin_t, x_shape) = rope_probe(seq, head_dim, base);

    // Host reference computed from the SAME rotate-half formula the recipe encodes.
    let x_vals = bytes_to_f32(&x.bytes);
    let cos_vals = bytes_to_f32(&cos_t.bytes);
    let sin_vals = bytes_to_f32(&sin_t.bytes);
    let expected = expected_rotate_half(&x_vals, &cos_vals, &sin_vals, seq, head_dim);

    let out = reference_from_registered_recipe(
        FusedOps::ROPE,
        &FusedOpParams::Rope,
        &[x, cos_t, sin_t],
        DType::F32,
        x_shape,
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    )
    .expect("reference_from_registered_recipe should realize the lowered rope base map");

    let got = bytes_to_f32(&out.bytes);
    assert_eq!(got.len(), expected.len(), "output element count");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-4,
            "rotate-half rope mismatch at {i}: got {g}, expected {e}"
        );
    }
}

/// Live-GPU: `reference_output` realizes a 2-input `Add` decompose region
/// on two F32 `[4]` probes and returns the elementwise sum. `#[ignore]`'d
/// (needs a live CUDA device); this is the Spec-B Task-4 acceptance test.
#[test]
#[ignore = "requires a live CUDA device"]
fn reference_output_realizes_the_decompose() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_cuda_backend::CudaDevice;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};

    let Ok(dev) = CudaDevice::new(0) else {
        return fuel_test_support::hardware::skip(
            fuel_test_support::hardware::Hardware::Cuda,
            fuel_test_support::hardware::Missing::device("CudaDevice::new(0) failed"),
        );
    };
    let region = PatternNode::Op {
        op: OpTag::Add,
        attrs: OpAttrs::default(),
        operands: vec![
            PatternNode::Bind { index: 0 },
            PatternNode::Bind { index: 1 },
        ],
    };
    let a = ht_f32(&[1.0, 2.0, 3.0, 4.0]);
    let b = ht_f32(&[10.0, 20.0, 30.0, 40.0]);
    let out = reference_output(
        &region,
        &[a, b],
        DType::F32,
        vec![4],
        &fuel_dispatch::probe_device::CudaProbeDevice::new(dev),
    )
    .unwrap();
    assert_eq!(bytes_to_f32(&out.bytes), vec![11.0, 22.0, 33.0, 44.0]);
}

/// **Where does `PagedAttn`'s DECOMPOSED form land on CUDA?** — the
/// measurement that gates whether offering a decomposed arm is worth
/// building, and what to ask Baracuda for.
///
/// Established already: the FUSED `Op::PagedAttn` node executes on the CPU
/// under CUDA decode (`paged_decode_node_placement_report_cuda`), because
/// no CUDA backend advertises it — so "runs on host" is not a decision the
/// planner made, it is the only move it had. `decompose` exists and is
/// closed-basis, but the planner is never offered it.
///
/// This asks the question that decides the next build, and it is a
/// GPU-vs-GPU question that the 186× host-ward figure cannot answer —
/// **any** on-device path removes that round-trip equally, so its magnitude
/// says something must be built without saying what:
///
/// - **every recipe node CUDA-placed** → the round-trip is removable with
///   NO kernel work at all; the ask to Baracuda may be nothing.
/// - **some primitive falls back to host** → *that primitive* is the ask,
///   far smaller and better-specified than "paged attention".
///
/// A grep cannot answer this. I tried: it reported `Permute` in zero
/// CUDA-backend files while the measured placement dump showed
/// `Permute Cuda×6` — it undercounts ops served by generic dispatch. Only
/// the planner's own resolved map settles it.
///
/// REPORTS rather than asserting a verdict, for the same reason as the
/// fused-node report: pre-committing to an answer is the error being
/// corrected. What it DOES assert is that the instrument ran
/// (`has_placements()`) and that decomposition actually happened (the root
/// moved off the fused node) — an all-`None` dump would read as "nothing
/// is on CUDA", a wrong answer wearing a null answer's clothes.
#[test]
#[ignore = "live GPU (CUDA); run via scripts/gpu-run.ps1"]
fn paged_attn_decomposed_arm_placement_report_cuda() {
    fuel_cuda_dispatch::install_global();
    fuel_test_support::require_gpu_run_lock();
    use fuel_graph::registry::{FusedOpParams, FusedOps};
    use std::collections::BTreeMap;

    const B: usize = 1;
    const HQ: usize = 4;
    const HKV: usize = 4;
    const SQ: usize = 1;
    const D: usize = 4;
    const NUM_BLOCKS: usize = 32;
    const BLOCK_SIZE: usize = 4;
    const MAX_BLK: usize = 8;

    let mut g = Graph::new();
    let leaf = |g: &mut Graph, dims: &[usize], dtype: DType| {
        g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: Shape::from_dims(dims),
            dtype,
        })
    };
    let q = leaf(&mut g, &[B, HQ, SQ, D], DType::F32);
    let kc = leaf(&mut g, &[NUM_BLOCKS, BLOCK_SIZE, HKV, D], DType::F32);
    let vc = leaf(&mut g, &[NUM_BLOCKS, BLOCK_SIZE, HKV, D], DType::F32);
    let bt = leaf(&mut g, &[B, MAX_BLK], DType::U32);
    let cl = leaf(&mut g, &[B], DType::U32);

    let params = FusedOpParams::PagedAttn {
        softmax_scale: 1.0 / (D as f32).sqrt(),
        block_size: BLOCK_SIZE,
        softcap: None,
    };
    let fused = g.push(Node {
        op: Op::Fused(FusedOps::PAGED_ATTN, params.clone()),
        inputs: vec![q, kc, vc, bt, cl],
        shape: Shape::from_dims(&[B, HQ, SQ, D]),
        dtype: DType::F32,
    });

    // Lower the fused node to its primitive recipe — the arm the planner is
    // never currently offered.
    let root = fuel_graph::registry::paged_attn::decompose(&mut g, fused, &params);
    assert_ne!(
        root, fused,
        "PRECONDITION: decompose must actually lower. A self-return is the \
         G2 fixpoint signal (malformed node / declined recipe), and every \
         placement below would then describe the FUSED node, silently \
         answering a different question than the one asked.",
    );

    // THE PRODUCTION TABLE, not a hand-assembled one. An earlier version of
    // this test built the table with `register_cuda_kernels` alone and got
    // "no backend supports index_select" — which was an artifact of the
    // HARNESS, not a fact about Fuel: that function makes ~5 registrations
    // and the bulk of CUDA lives in `register_baracuda_cuda_kernels`, which
    // the real path (`register_optional_backends`) also calls, on top of
    // `register_cpu_kernels` and a cost fill. Reproducing that assembly by
    // hand is precisely the kind of thing that silently drifts, so ask for
    // the table the realize path actually uses.
    let table = fuel_dispatch::dispatch::global_bindings();

    // POSITIVE CONTROL — the instrument must be able to see a CUDA binding
    // at all. Without this, an under-populated table reports "no CUDA
    // support" for EVERY op, which reads exactly like a real finding: it is
    // specific, it names a primitive, and it is completely wrong. That is
    // how the first run of this test nearly shipped a false Baracuda ask.
    // MatMul is the control because the fuel-core placement report measured
    // `MatMul Cuda×15` on a real decode graph — an independently observed
    // fact this table must reproduce.
    assert!(
        table
            .lookup(
                OpKind::MatMul,
                &[DType::F32, DType::F32, DType::F32],
                BackendId::Cuda
            )
            .is_ok(),
        "INSTRUMENT BROKEN: the binding table has no CUDA MatMul, so every \
         'no CUDA support' conclusion below would be an artifact of the \
         harness. Fix the table before reading any verdict from this test.",
    );

    // ---- THE COMPLETE MISSING-BINDING SET, asked directly ----
    //
    // `optimize_graph` fails at the FIRST unplaceable node, so driving this
    // off its error yields one op per run — a trickle, and each run costs a
    // CUDA rebuild. Worse, it conflates two different things: "CUDA has no
    // kernel for this op" and "the planner could not place it under THIS
    // pin". Asking the binding table per node separates them and returns the
    // whole set at once.
    let mut missing_cuda: Vec<String> = Vec::new();
    let mut present_cuda: Vec<String> = Vec::new();
    for i in 0..g.len() {
        let id = NodeId(i);
        let node = g.node(id);
        let Some(kind) = op_to_op_kind(&node.op) else {
            continue; // Const / view / fused — not a binding-table lookup
        };
        let mut dtypes: Vec<DType> = node.inputs.iter().map(|&i| g.node(i).dtype).collect();
        dtypes.push(node.dtype); // binding key is [inputs.., output]
        let sig = format!(
            "{kind:?}{:?}",
            dtypes.iter().map(|d| format!("{d:?}")).collect::<Vec<_>>()
        );
        if table.lookup(kind, &dtypes, BackendId::Cuda).is_ok() {
            present_cuda.push(sig);
        } else {
            missing_cuda.push(sig);
        }
    }
    missing_cuda.sort();
    missing_cuda.dedup();
    present_cuda.sort();
    present_cuda.dedup();

    println!("\n=== PagedAttn recipe: CUDA binding coverage (asked per node) ===");
    println!("PRESENT on CUDA ({}):", present_cuda.len());
    for s in &present_cuda {
        println!("    {s}");
    }
    println!("MISSING on CUDA ({}):", missing_cuda.len());
    for s in &missing_cuda {
        println!("    {s}");
    }
    // A run where NOTHING resolves would mean the table is empty again — the
    // MatMul control above already guards that, but state the invariant that
    // makes the "missing" list meaningful rather than vacuous.
    assert!(
        !present_cuda.is_empty(),
        "INSTRUMENT BROKEN: not one recipe node resolved on CUDA, so the \
         MISSING list is a property of the lookup, not of the backend",
    );
    println!(
        "\n>>> THE ASK: the {} signature(s) above are what CUDA lacks for the \
         §1 dense recipe. Each is a small, named primitive — not \"paged \
         attention\".\n",
        missing_cuda.len()
    );

    let opts = PlanOptions::new().with_pinned_device(DeviceLocation::Cuda { gpu_id: 0 });
    let optimized = match optimize_graph(&mut g, &[root], &table, &opts) {
        Ok(o) => o,
        Err(e) => {
            println!("\n=== DECOMPOSED ARM DOES NOT PLAN ON CUDA ===");
            println!("optimize_graph error: {e:?}");
            println!("That is itself the answer: some recipe primitive has no CUDA");
            println!("binding, and the error names it. THAT is the Baracuda ask.");
            return;
        }
    };

    assert!(
        optimized.has_placements(),
        "placement NOT COMPUTED — every placement_of would be None and the \
         report below would describe the missing instrument, not the graph",
    );

    let mut by_op: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut host_ops: Vec<String> = Vec::new();
    for i in 0..g.len() {
        let id = NodeId(i);
        let op_name = match &g.node(id).op {
            Op::Fused(fid, _) => format!("Fused({fid:?})"),
            other => format!("{other:?}")
                .split(' ')
                .next()
                .unwrap_or("?")
                .to_string(),
        };
        let place = match optimized.placement_of(id) {
            Some(d) => format!("{d:?}"),
            None => "None".to_string(),
        };
        if place.starts_with("Cpu") {
            host_ops.push(op_name.clone());
        }
        *by_op.entry(op_name).or_default().entry(place).or_insert(0) += 1;
    }

    println!("\n=== PagedAttn DECOMPOSED arm: node placement (CUDA-pinned) ===");
    println!(
        "total nodes: {} (fused node #{} lowered to root #{})",
        g.len(),
        fused.0,
        root.0
    );
    for (op, places) in &by_op {
        let rendered: Vec<String> = places.iter().map(|(d, n)| format!("{d}×{n}")).collect();
        println!("  {op:<40} {}", rendered.join("  "));
    }

    println!("\n--- VERDICT ---");
    if host_ops.is_empty() {
        println!(
            "EVERY recipe node is off-host. The decomposed paged attention runs \
             ON DEVICE, so the 186x round-trip is removable by OFFERING THIS ARM \
             — no Baracuda kernel required to eliminate it. A kernel would then \
             be a throughput question (the recipe materialises padded capacity, \
             not live context_len), not a round-trip one."
        );
    } else {
        host_ops.sort();
        host_ops.dedup();
        println!(
            "HOST-PLACED recipe primitives: {host_ops:?}\n\
             These are the ask — each is a far smaller and better-specified \
             request than \"paged attention\", and until they exist on device \
             the decomposed arm cannot win."
        );
    }
    println!("--- END VERDICT ---\n");
}
