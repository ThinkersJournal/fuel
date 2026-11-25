// SPDX-License-Identifier: MIT OR Apache-2.0
//! The conv epilogue (ROADMAP row 172 item 3; design:
//! `docs/design-notes/2026-11-25-row172-item3-conv-epilogue.md`, revision at the top).
//!
//! The epilogue is an ELEMENTWISE RECIPE over the conv output, not a list of
//! activations: [`extract_epilogue`] reads, from a root, the elementwise region that
//! consumes a `CONV2D` node and returns it as portable recipe data (a
//! [`PatternNode`] whose `Bind 0` is the conv output, `Bind 1..` the extra operands:
//! bias broadcast, scale, residual) plus the live scalars in pre-order, the same open-slot
//! convention every other recipe uses. Anything not elementwise is an operand, not part of
//! the region.

use crate::registry::{FusedOpParams, FusedOps};
use crate::{Graph, NodeId, Op};
use fireline_grammar::{OpAttrs, OpTag, PatternNode};
use std::collections::HashMap;

/// What [`extract_epilogue`] found above a conv.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    /// The `CONV2D` node the region consumes.
    pub conv: NodeId,
    /// The recipe: `Bind 0` = the conv output, `Bind 1..=extras.len()` = [`Self::extras`].
    pub recipe: PatternNode,
    /// The live scalars of the recipe's open slots, in pre-order.
    pub scalars: Vec<f64>,
    /// The non-elementwise operands (bias broadcast, scale, residual), by `Bind` index
    /// `1..`.
    pub extras: Vec<NodeId>,
}

/// Elementwise ops an epilogue region may contain (a property of the op, not a list of
/// activations: every one of these is a primitive already in the graph's op set).
fn epilogue_tag(op: &Op) -> Option<OpTag> {
    let tag = crate::jit::op_to_tag(op)?;
    use OpTag::*;
    matches!(
        tag,
        Add | Sub
            | Mul
            | Div
            | Maximum
            | Minimum
            | Neg
            | Abs
            | Sqr
            | Sqrt
            | Rsqrt
            | Recip
            | Exp
            | Log
            | Sin
            | Cos
            | Tanh
            | Sigmoid
            | Silu
            | GeluTanh
            | Gelu
            | Relu
            | Erf
            | Step
            | Floor
            | Ceil
            | Round
            | Sign
            | AddScalar
            | MulScalar
            | Clamp
            | Equal
            | Ne
            | Lt
            | Le
            | Gt
            | Ge
            | Where
    )
    .then_some(tag)
}

fn scalars_of(op: &Op) -> Vec<f64> {
    match op {
        Op::AddScalar(c) | Op::MulScalar(c) => vec![*c],
        Op::Clamp { min, max } => vec![*min, *max],
        _ => Vec::new(),
    }
}

fn is_conv(graph: &Graph, id: NodeId) -> bool {
    matches!(
        graph.node(id).op,
        Op::Fused(fid, FusedOpParams::Conv2D { .. }) if fid == FusedOps::CONV2D
    )
}

/// Read the elementwise region rooted at `root` that consumes exactly one `CONV2D`
/// node. `None` when there is no conv under `root`, no elementwise op between them, or
/// the conv's output is also consumed OUTSIDE the region (fusing would compute it twice).
pub fn extract_epilogue(graph: &Graph, root: NodeId) -> Option<Extracted> {
    struct Walk<'a> {
        graph: &'a Graph,
        conv: Option<NodeId>,
        scalars: Vec<f64>,
        extras: Vec<NodeId>,
        in_region: HashMap<NodeId, usize>, // region node -> references made to it
        conv_refs: usize,
        elementwise: usize,
        mixed: bool, // a second, different conv was reached
    }
    impl Walk<'_> {
        fn visit(&mut self, id: NodeId) -> PatternNode {
            if is_conv(self.graph, id) {
                match self.conv {
                    None => self.conv = Some(id),
                    Some(c) if c != id => self.mixed = true,
                    Some(_) => {}
                }
                self.conv_refs += 1;
                return PatternNode::Bind { index: 0 };
            }
            let node = self.graph.node(id);
            if let Some(tag) = epilogue_tag(&node.op) {
                self.elementwise += 1;
                *self.in_region.entry(id).or_insert(0) += 1;
                // Pre-order: this node's open scalar slots come before its children's.
                self.scalars.extend(scalars_of(&node.op));
                let operands: Vec<PatternNode> =
                    node.inputs.iter().map(|&c| self.visit(c)).collect();
                return PatternNode::Op {
                    op: tag,
                    operands,
                    attrs: OpAttrs::default(),
                };
            }
            // Not elementwise and not the conv: an operand of the region (deduplicated
            // by node identity, so a repeated operand is the same Bind index).
            let slot = match self.extras.iter().position(|&e| e == id) {
                Some(k) => k,
                None => {
                    self.extras.push(id);
                    self.extras.len() - 1
                }
            };
            PatternNode::Bind {
                index: u8::try_from(slot + 1).unwrap_or(u8::MAX),
            }
        }
    }
    let mut w = Walk {
        graph,
        conv: None,
        scalars: Vec::new(),
        extras: Vec::new(),
        in_region: HashMap::new(),
        conv_refs: 0,
        elementwise: 0,
        mixed: false,
    };
    let recipe = w.visit(root);
    let conv = w.conv?;
    if w.mixed || w.elementwise == 0 || w.extras.len() >= usize::from(u8::MAX) {
        return None;
    }
    // Every consumer of the conv must be inside the region: the fused node replaces the
    // conv, so a consumer outside would force the conv to be computed a second time.
    let counts = consumer_counts(graph);
    if counts.get(&conv).copied().unwrap_or(0)
        != distinct_region_consumers(graph, conv, &w.in_region)
    {
        return None;
    }
    // Likewise every interior node (all but the root) must be read only inside the region.
    for (&n, &refs) in &w.in_region {
        if n != root && counts.get(&n).copied().unwrap_or(0) != refs {
            return None;
        }
    }
    Some(Extracted {
        conv,
        recipe,
        scalars: w.scalars,
        extras: w.extras,
    })
}

/// How many (consumer node, input slot) references to `conv` come from region nodes.
fn distinct_region_consumers(
    graph: &Graph,
    conv: NodeId,
    region: &HashMap<NodeId, usize>,
) -> usize {
    region
        .keys()
        .map(|&n| graph.node(n).inputs.iter().filter(|&&i| i == conv).count())
        .sum()
}

fn consumer_counts(graph: &Graph) -> HashMap<NodeId, usize> {
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
    use crate::{Node, NodeHandle};
    use fuel_ir::{DType, Shape};
    use std::sync::{Arc, RwLock};

    fn conv_graph() -> (Arc<RwLock<Graph>>, NodeHandle) {
        let graph = Arc::new(RwLock::new(Graph::new()));
        let (x, w) = {
            let mut g = graph.write().unwrap();
            let c = |g: &mut Graph, dims: &[usize]| {
                g.push(Node {
                    op: Op::Const,
                    inputs: vec![],
                    shape: Shape::from_dims(dims),
                    dtype: DType::F32,
                })
            };
            (c(&mut g, &[1, 2, 6, 6]), c(&mut g, &[3, 2, 3, 3]))
        };
        let x = NodeHandle::from_existing(Arc::clone(&graph), x);
        let w = NodeHandle::from_existing(Arc::clone(&graph), w);
        let conv = x.conv2d(&w, None, (1, 1), (1, 1), 1);
        (graph, conv)
    }

    fn op(tag: OpTag, operands: Vec<PatternNode>) -> PatternNode {
        PatternNode::Op {
            op: tag,
            operands,
            attrs: OpAttrs::default(),
        }
    }
    fn bind(i: u8) -> PatternNode {
        PatternNode::Bind { index: i }
    }

    #[test]
    fn a_pure_activation_is_the_whole_recipe() {
        let (graph, conv) = conv_graph();
        let out = conv.relu();
        let g = graph.read().unwrap();
        let e = extract_epilogue(&g, out.id()).expect("relu over a conv is an epilogue");
        assert_eq!(e.conv, conv.id());
        assert_eq!(e.recipe, op(OpTag::Relu, vec![bind(0)]));
        assert!(e.scalars.is_empty() && e.extras.is_empty());
    }

    /// leaky-relu = max(x, 0.2 * x): the conv output is read twice, the scalar is carried
    /// in pre-order (the open-slot convention), no extra operand.
    #[test]
    fn a_shared_conv_output_and_a_scalar_are_carried() {
        let (graph, conv) = conv_graph();
        let out = conv.maximum(&conv.mul_scalar(0.2));
        let g = graph.read().unwrap();
        let e = extract_epilogue(&g, out.id()).expect("leaky relu");
        assert_eq!(
            e.recipe,
            op(
                OpTag::Maximum,
                vec![bind(0), op(OpTag::MulScalar, vec![bind(0)])]
            )
        );
        assert_eq!(e.scalars, vec![0.2]);
        assert!(e.extras.is_empty());
    }

    /// conv + a broadcast bias then an activation: the broadcast bias is an extra operand
    /// (Bind 1), not part of the region (it is not elementwise).
    #[test]
    fn a_bias_broadcast_is_an_extra_operand() {
        let (graph, conv) = conv_graph();
        let bias = {
            let mut g = graph.write().unwrap();
            let b = g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(&[3]),
                dtype: DType::F32,
            });
            let bb = g.push(Node {
                op: Op::BroadcastTo(Shape::from_dims(&[1, 3, 6, 6])),
                inputs: vec![b],
                shape: Shape::from_dims(&[1, 3, 6, 6]),
                dtype: DType::F32,
            });
            NodeHandle::from_existing(Arc::clone(&graph), bb)
        };
        let out = conv.add(&bias).tanh();
        let g = graph.read().unwrap();
        let e = extract_epilogue(&g, out.id()).expect("bias + tanh");
        assert_eq!(
            e.recipe,
            op(OpTag::Tanh, vec![op(OpTag::Add, vec![bind(0), bind(1)])])
        );
        assert_eq!(e.extras, vec![bias.id()]);
    }

    /// An interior node also read outside the region would stay in the graph and keep the
    /// conv alive: the region is declined.
    #[test]
    fn declines_when_an_interior_node_is_read_outside_the_region() {
        let (graph, conv) = conv_graph();
        let mid = conv.relu();
        let out = mid.tanh();
        let _other = mid.neg(); // `mid` is also consumed outside the region
        let g = graph.read().unwrap();
        assert!(extract_epilogue(&g, out.id()).is_none());
    }

    /// Sabotage siblings: declines with no conv, with nothing elementwise above the conv,
    /// and when the conv output is ALSO consumed outside the region.
    #[test]
    fn declines_without_a_conv_a_region_or_when_the_conv_is_shared() {
        let (graph, conv) = conv_graph();
        let x = {
            let g = graph.read().unwrap();
            g.node(conv.id()).inputs[0]
        };
        let g = graph.read().unwrap();
        assert!(
            extract_epilogue(&g, x).is_none(),
            "a Const root has no conv"
        );
        assert!(
            extract_epilogue(&g, conv.id()).is_none(),
            "the conv alone is no region"
        );
        drop(g);
        let out = conv.relu();
        let _other = conv.exp(); // the conv output is also read outside the region
        let g = graph.read().unwrap();
        assert!(
            extract_epilogue(&g, out.id()).is_none(),
            "a shared conv is not fused"
        );
    }
}
