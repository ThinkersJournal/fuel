// SPDX-License-Identifier: MIT OR Apache-2.0
//! RowScaledMatMul: `(a @ b) * row_scale`, `row_scale` of shape `[..batch, M, 1]`
//! (a per-row / per-token scale), as a fused op (ROADMAP row 172 item 1;
//! design: `docs/design-notes/2026-11-25-row172-item1-row-scaled-matmul.md`).
//!
//! Recipe (portable data, parameterless):
//! `Mul(MatMul(a, b), BroadcastTo(WithDim{op0, LAST, Extent{op1, LAST}})(scale))`.
//! Binds: `0 = a`, `1 = b`, `2 = scale`.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId, Op};
use fireline_grammar::shape_expr::{Dim, LAST, ShapeExpr};
use fireline_grammar::{OpAttrs, OpTag, PatternNode};
use fuel_ir::{DType, Shape};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Metadata-side registry entry for RowScaledMatMul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::ROW_SCALED_MATMUL,
        name: "RowScaledMatMul",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::NotDifferentiable,
        shape_rule: matmul_output_shape,
        dtype_rule: dtype_passthrough,
        output_views: None,
    }
}

/// Output shape: the matmul output `[..batch, M, N]`; the scale is
/// `[..batch, M, 1]` and broadcasts along N.
fn matmul_output_shape(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(input_shapes.len(), 3, "RowScaledMatMul takes (a, b, scale)");
    let a = &input_shapes[0];
    let b = &input_shapes[1];
    let (a_rank, b_rank) = (a.rank(), b.rank());
    debug_assert!(a_rank >= 2 && a_rank == b_rank);
    let mut dims: Vec<usize> = a.dims()[..a_rank - 2].to_vec();
    dims.push(a.dims()[a_rank - 2]); // M
    dims.push(b.dims()[b_rank - 1]); // N
    Shape::from_dims(&dims)
}

fn dtype_passthrough(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 3, "RowScaledMatMul takes 3 inputs");
    input_dtypes[0]
}

/// The primitive recipe as data. The scale's last extent is 1, so the
/// `WithDim` broadcast target (a's shape with K replaced by N) expands it
/// along N with no reshape.
fn recipe() -> &'static PatternNode {
    static RECIPE: OnceLock<PatternNode> = OnceLock::new();
    RECIPE.get_or_init(|| {
        let op = |op, attrs, operands| PatternNode::Op {
            op,
            attrs,
            operands,
        };
        let bind = |index| PatternNode::Bind { index };
        let target = OpAttrs {
            target_shape_rel: Some(ShapeExpr::WithDim {
                operand: 0,
                axis: LAST,
                dim: Box::new(Dim::Extent {
                    operand: 1,
                    axis: LAST,
                }),
            }),
            ..OpAttrs::default()
        };
        op(
            OpTag::Mul,
            OpAttrs::default(),
            vec![
                op(OpTag::MatMul, OpAttrs::default(), vec![bind(0), bind(1)]),
                op(OpTag::BroadcastTo, target, vec![bind(2)]),
            ],
        )
    })
}

fn scalars(params: &FusedOpParams) -> Option<Vec<f64>> {
    match params {
        FusedOpParams::RowScaledMatMul => Some(Vec::new()),
        _ => None,
    }
}

/// Lower a fused RowScaledMatMul node to `Mul(MatMul(a, b), BroadcastTo(scale))`.
/// Any failure returns `id` unchanged (G2; never a panic).
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    decompose_via_recipe(graph, id, recipe(), scalars(params))
}

/// Match `Mul(MatMul(a, b), BroadcastTo(scale))` (either operand order) where
/// `scale` is `[..batch, M, 1]`: same rank as the matmul output, last extent 1,
/// every other extent equal to the matmul output's. The MatMul must have the
/// Mul as its only consumer. A `[N]` bias-shaped operand does not match.
pub fn canonical_pattern(graph: &Graph, mul_id: NodeId) -> Option<PatternMatch> {
    let mul = graph.node(mul_id);
    if !matches!(mul.op, Op::Mul) || mul.inputs.len() != 2 {
        return None;
    }
    // Mul is commutative: the MatMul may be either operand.
    let (mm_id, bc_id) = if matches!(graph.node(mul.inputs[0]).op, Op::MatMul) {
        (mul.inputs[0], mul.inputs[1])
    } else {
        (mul.inputs[1], mul.inputs[0])
    };
    let mm = graph.node(mm_id);
    if !matches!(mm.op, Op::MatMul) || mm.inputs.len() != 2 {
        return None;
    }
    let bc = graph.node(bc_id);
    if !matches!(bc.op, Op::BroadcastTo(_)) || bc.inputs.len() != 1 {
        return None;
    }
    let scale_id = bc.inputs[0];
    let out_dims = mm.shape.dims();
    let scale_dims = graph.node(scale_id).shape.dims();
    // `[..batch, M, 1]`: the matmul output's shape with N replaced by 1. This
    // also rejects a `[N]` bias, a `[.., 1, N]` column scale and a full-shape
    // operand (a different fusion, an elementwise Mul).
    if out_dims.len() < 2
        || scale_dims.len() != out_dims.len()
        || scale_dims[scale_dims.len() - 1] != 1
        || scale_dims[..scale_dims.len() - 1] != out_dims[..out_dims.len() - 1]
        || out_dims[out_dims.len() - 1] == 1
        || bc.shape.dims() != out_dims
    {
        return None;
    }
    let counts = count_consumers(graph);
    if counts.get(&mm_id).copied().unwrap_or(0) != 1 {
        return None;
    }
    Some(PatternMatch {
        bindings: vec![(0, mm.inputs[0]), (1, mm.inputs[1]), (2, scale_id)],
        params: FusedOpParams::RowScaledMatMul,
    })
}

fn count_consumers(graph: &Graph) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for nid in 0..graph.len() {
        for &input in &graph.node(NodeId(nid)).inputs {
            *counts.entry(input).or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Node;

    fn f32_node(op: Op, inputs: Vec<NodeId>, dims: &[usize]) -> Node {
        Node {
            op,
            inputs,
            shape: Shape::from_dims(dims),
            dtype: DType::F32,
        }
    }

    /// `Mul(MatMul(a[M,K], b[K,N]), BroadcastTo(scale))`, scale shaped `scale_dims`.
    fn scaled(g: &mut Graph, scale_dims: &[usize], swap: bool) -> (NodeId, [NodeId; 3], NodeId) {
        let a = g.push(f32_node(Op::Const, vec![], &[2, 3, 8, 16]));
        let b = g.push(f32_node(Op::Const, vec![], &[2, 3, 16, 32]));
        let s = g.push(f32_node(Op::Const, vec![], scale_dims));
        let mm = g.push(f32_node(Op::MatMul, vec![a, b], &[2, 3, 8, 32]));
        let bc = g.push(f32_node(
            Op::BroadcastTo(Shape::from_dims(&[2, 3, 8, 32])),
            vec![s],
            &[2, 3, 8, 32],
        ));
        let ins = if swap { vec![bc, mm] } else { vec![mm, bc] };
        let mul = g.push(f32_node(Op::Mul, ins, &[2, 3, 8, 32]));
        (mul, [a, b, s], mm)
    }

    #[test]
    fn matches_a_per_row_scale_in_both_operand_orders() {
        for swap in [false, true] {
            let mut g = Graph::new();
            let (mul, [a, b, s], _) = scaled(&mut g, &[2, 3, 8, 1], swap);
            let m = canonical_pattern(&g, mul).expect("per-row scale must match");
            assert_eq!(m.bindings, vec![(0, a), (1, b), (2, s)], "swap={swap}");
            assert!(matches!(m.params, FusedOpParams::RowScaledMatMul));
        }
    }

    /// Sabotage siblings: a `[N]`-shaped (bias-like) or `[.., 1, N]` operand is a
    /// column scale, not a row scale, and must not match.
    #[test]
    fn declines_a_column_shaped_operand() {
        for dims in [&[32usize][..], &[2, 3, 1, 32][..], &[2, 3, 8, 32][..]] {
            let mut g = Graph::new();
            let (mul, _, _) = scaled(&mut g, dims, false);
            assert!(canonical_pattern(&g, mul).is_none(), "scale {dims:?}");
        }
    }

    #[test]
    fn declines_when_the_matmul_has_another_consumer() {
        let mut g = Graph::new();
        let (mul, _, mm) = scaled(&mut g, &[2, 3, 8, 1], false);
        let _other = g.push(f32_node(Op::Relu, vec![mm], &[2, 3, 8, 32]));
        assert!(canonical_pattern(&g, mul).is_none());
    }

    /// Lower then fuse recovers the node (the framework's own emission re-fuses).
    #[test]
    fn decompose_then_pattern_round_trips() {
        let mut g = Graph::new();
        let a = g.push(f32_node(Op::Const, vec![], &[2, 3, 8, 16]));
        let b = g.push(f32_node(Op::Const, vec![], &[2, 3, 16, 32]));
        let s = g.push(f32_node(Op::Const, vec![], &[2, 3, 8, 1]));
        let fused = g.push(f32_node(
            Op::Fused(FusedOps::ROW_SCALED_MATMUL, FusedOpParams::RowScaledMatMul),
            vec![a, b, s],
            &[2, 3, 8, 32],
        ));
        let root = decompose(&mut g, fused, &FusedOpParams::RowScaledMatMul);
        assert_ne!(root, fused, "the recipe decompose fires");
        let m = canonical_pattern(&g, root).expect("the recipe emission re-fuses");
        assert_eq!(m.bindings, vec![(0, a), (1, b), (2, s)]);
    }
}
