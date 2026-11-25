// SPDX-License-Identifier: MIT OR Apache-2.0
//! MatMulTopK: `TopKLastDim(a @ b)` as ONE fused node, the TopK epilogue on the
//! matmul (ROADMAP row 172 item 2 step B; design:
//! `docs/design-notes/2026-11-25-row172-item2-matmul-topk-epilogue.md`).
//!
//! Inputs `(a [.., M, K], b [.., K, N])` with equal batch dims; the matmul output
//! `[.., M, N]` never materializes. Outputs: the `TopKLastDim` bundle over `N`
//! (slot 0 values `[.., M, k]`, slot 1 `U32` indices `[.., M, k]`).
//!
//! Recognized from `TopKLastDim(MatMul(a, b))` where the MatMul has no other
//! consumer; `decompose` is that two-node form.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern,
};
use crate::{Graph, Node, NodeId, Op};
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Shape};
use std::collections::HashMap;

/// Metadata-side registry entry for MatMulTopK (multi-output, like TopKLastDim).
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::MATMUL_TOP_K,
        name: "MatMulTopK",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: Some(output_views),
    }
}

fn k_of(params: &FusedOpParams) -> usize {
    match params {
        FusedOpParams::MatMulTopK { k } => *k,
        _ => 0,
    }
}

/// Slot-0 shape: `a`'s batch dims ++ `[M, k]`.
pub fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    let a = input_shapes[0].dims();
    let mut dims = a[..a.len().saturating_sub(1)].to_vec();
    dims.push(k_of(params));
    Shape::from_dims(&dims)
}

/// Slot-0 dtype: `a`'s.
pub fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    input_dtypes[0]
}

/// `values` (input dtype) then `indices` (`U32`), both `[.., M, k]`.
pub fn output_views(
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    params: &FusedOpParams,
) -> Vec<OutputViewSpec> {
    let shape = shape_rule(input_shapes, params);
    vec![
        OutputViewSpec::contiguous(input_dtypes[0], shape.clone()),
        OutputViewSpec::contiguous(DType::U32, shape),
    ]
}

/// Lower to `TopKLastDim(MatMul(a, b))`, re-attaching the bundle contract to
/// the new TopK node so existing `View` consumers stay valid. Returns `id`
/// unchanged on any malformed shape (G2; never a panic).
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let FusedOpParams::MatMulTopK { k } = params else {
        return id;
    };
    let (a, b, dtype, shape, views) = {
        let n = graph.node(id);
        if n.inputs.len() != 2 {
            return id;
        }
        (
            n.inputs[0],
            n.inputs[1],
            n.dtype,
            n.shape.clone(),
            graph.output_views_arc(id),
        )
    };
    let (a_dims, b_dims) = (graph.node(a).shape.dims(), graph.node(b).shape.dims());
    if a_dims.len() < 2 || a_dims.len() != b_dims.len() {
        return id;
    }
    let mut mm_dims = a_dims[..a_dims.len() - 1].to_vec();
    mm_dims.push(b_dims[b_dims.len() - 1]);
    let mm = graph.push(Node {
        op: Op::MatMul,
        inputs: vec![a, b],
        shape: Shape::from_dims(&mm_dims),
        dtype,
    });
    let top = graph.push(Node {
        op: Op::Fused(
            FusedOps::TOP_K_LAST_DIM,
            FusedOpParams::TopKLastDim { k: *k },
        ),
        inputs: vec![mm],
        shape,
        dtype,
    });
    if let Some(v) = views
        && graph.set_output_views(top, v).is_err()
    {
        return id;
    }
    top
}

/// Match a `TopKLastDim` node whose input is a single-consumer `MatMul` with
/// equal batch dims. Bindings `[(0, a), (1, b)]`, params `MatMulTopK { k }`.
pub fn canonical_pattern(graph: &Graph, top_id: NodeId) -> Option<PatternMatch> {
    let top = graph.node(top_id);
    let Op::Fused(fid, FusedOpParams::TopKLastDim { k }) = &top.op else {
        return None;
    };
    if *fid != FusedOps::TOP_K_LAST_DIM || top.inputs.len() != 1 {
        return None;
    }
    let mm_id = top.inputs[0];
    let mm = graph.node(mm_id);
    if !matches!(mm.op, Op::MatMul) || mm.inputs.len() != 2 {
        return None;
    }
    let (a, b) = (mm.inputs[0], mm.inputs[1]);
    let (ad, bd) = (graph.node(a).shape.dims(), graph.node(b).shape.dims());
    if ad.len() < 2 || ad.len() != bd.len() || ad[..ad.len() - 2] != bd[..bd.len() - 2] {
        return None; // equal batch dims only (no GQA broadcast in the fused kernel)
    }
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for nid in 0..graph.len() {
        for &input in &graph.node(NodeId(nid)).inputs {
            *counts.entry(input).or_insert(0) += 1;
        }
    }
    if counts.get(&mm_id).copied().unwrap_or(0) != 1 {
        return None;
    }
    Some(PatternMatch {
        bindings: vec![(0, a), (1, b)],
        params: FusedOpParams::MatMulTopK { k: *k },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_node(op: Op, inputs: Vec<NodeId>, dims: &[usize]) -> Node {
        Node {
            op,
            inputs,
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        }
    }

    fn topk_over_matmul(
        g: &mut Graph,
        a_dims: &[usize],
        b_dims: &[usize],
        k: usize,
    ) -> (NodeId, [NodeId; 2], NodeId) {
        let a = g.push(f32_node(Op::Const, vec![], a_dims));
        let b = g.push(f32_node(Op::Const, vec![], b_dims));
        let mut out = a_dims[..a_dims.len() - 1].to_vec();
        out.push(b_dims[b_dims.len() - 1]);
        let mm = g.push(f32_node(Op::MatMul, vec![a, b], &out));
        let mut top_dims = out.clone();
        *top_dims.last_mut().unwrap() = k;
        let top = g.push(f32_node(
            Op::Fused(FusedOps::TOP_K_LAST_DIM, FusedOpParams::TopKLastDim { k }),
            vec![mm],
            &top_dims,
        ));
        (top, [a, b], mm)
    }

    #[test]
    fn matches_topk_over_a_single_consumer_matmul() {
        let mut g = Graph::new();
        let (top, [a, b], _) = topk_over_matmul(&mut g, &[2, 4, 8], &[2, 8, 16], 3);
        let m = canonical_pattern(&g, top).expect("TopK(MatMul) fuses");
        assert_eq!(m.bindings, vec![(0, a), (1, b)]);
        assert!(matches!(m.params, FusedOpParams::MatMulTopK { k: 3 }));
    }

    /// Sabotage siblings: a shared MatMul, a GQA batch mismatch, and a non-TopK
    /// root must not match.
    #[test]
    fn declines_a_shared_matmul_a_batch_mismatch_and_other_roots() {
        let mut g = Graph::new();
        let (top, _, mm) = topk_over_matmul(&mut g, &[2, 4, 8], &[2, 8, 16], 3);
        let _other = g.push(f32_node(Op::Relu, vec![mm], &[2, 4, 16]));
        assert!(canonical_pattern(&g, top).is_none(), "shared matmul");
        let mut g = Graph::new();
        let (top, _, _) = topk_over_matmul(&mut g, &[4, 4, 8], &[2, 8, 16], 3);
        assert!(canonical_pattern(&g, top).is_none(), "batch mismatch");
        let mut g = Graph::new();
        let (_, _, mm) = topk_over_matmul(&mut g, &[4, 8], &[8, 16], 3);
        assert!(canonical_pattern(&g, mm).is_none(), "a MatMul root");
    }

    #[test]
    fn slot_shapes_are_batch_m_by_k() {
        let p = FusedOpParams::MatMulTopK { k: 3 };
        let ins = [Shape::from_dims(&[2, 4, 8]), Shape::from_dims(&[2, 8, 16])];
        assert_eq!(shape_rule(&ins, &p).dims(), &[2, 4, 3]);
        let v = output_views(&ins, &[DType::F32, DType::F32], &p);
        assert_eq!((v[0].dtype, v[1].dtype), (DType::F32, DType::U32));
        assert_eq!(v[1].shape.dims(), &[2, 4, 3]);
    }

    /// The generic fusion rule fires on a builder-made `TopKLastDim(MatMul)` and
    /// keeps the multi-output bundle contract on the replacement node (the rule's
    /// first multi-output root), so the `View` consumers still resolve both slots.
    #[test]
    fn the_fusion_rule_rewrites_a_multi_output_root_and_keeps_its_views() {
        use crate::NodeHandle;
        use crate::opt::{FusionRule, Rule};
        use std::sync::{Arc, RwLock};
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (a_id, b_id) = {
            let mut g = graph.write().unwrap();
            (
                g.push(f32_node(Op::Const, vec![], &[2, 4, 8])),
                g.push(f32_node(Op::Const, vec![], &[2, 8, 16])),
            )
        };
        let a = NodeHandle::from_existing(Arc::clone(&graph), a_id);
        let b = NodeHandle::from_existing(Arc::clone(&graph), b_id);
        let (vals, idx) = a.matmul(&b).top_k_last_dim(3).expect("builder");
        let rule = FusionRule::from_entry(&entry());
        let mut g = graph.write().unwrap();
        let top = g.node(vals.id()).inputs[0];
        assert_eq!(
            g.node(idx.id()).inputs[0],
            top,
            "both views project the TopK node"
        );
        assert!(rule.matches(&g, top));
        let mut remap = HashMap::new();
        rule.rewrite(&mut g, top, &mut remap);
        let fused = remap[&top];
        assert!(matches!(
            g.node(fused).op,
            Op::Fused(id, FusedOpParams::MatMulTopK { k: 3 }) if id == FusedOps::MATMUL_TOP_K
        ));
        assert_eq!(g.node(fused).inputs, vec![a_id, b_id]);
        let (old, new) = (g.output_views(top).unwrap(), g.output_views(fused).unwrap());
        assert_eq!(old.len(), 2);
        let key = |v: &[fuel_ir::storage::OutputView]| {
            v.iter()
                .map(|s| (s.byte_offset, s.len_elements, s.dtype))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            key(old),
            key(new),
            "the bundle contract moved with the node"
        );
    }

    /// decompose yields TopKLastDim(MatMul) and keeps the bundle contract; the
    /// recognizer then re-fuses it (lower -> fuse round-trip).
    #[test]
    fn decompose_then_pattern_round_trips() {
        let mut g = Graph::new();
        let a = g.push(f32_node(Op::Const, vec![], &[2, 4, 8]));
        let b = g.push(f32_node(Op::Const, vec![], &[2, 8, 16]));
        let params = FusedOpParams::MatMulTopK { k: 3 };
        let specs = output_views(
            &[Shape::from_dims(&[2, 4, 8]), Shape::from_dims(&[2, 8, 16])],
            &[DType::F32, DType::F32],
            &params,
        );
        let (_, views) = fuel_ir::storage::compose_bundle(&specs).unwrap();
        let fused = g.push(f32_node(
            Op::Fused(FusedOps::MATMUL_TOP_K, params.clone()),
            vec![a, b],
            &[2, 4, 3],
        ));
        g.set_output_views(fused, std::sync::Arc::from(views.into_boxed_slice()))
            .unwrap();
        let root = decompose(&mut g, fused, &params);
        assert_ne!(root, fused);
        assert!(
            g.output_views(root).is_some(),
            "bundle contract re-attached"
        );
        let m = canonical_pattern(&g, root).expect("re-fuses");
        assert_eq!(m.bindings, vec![(0, a), (1, b)]);
    }
}
