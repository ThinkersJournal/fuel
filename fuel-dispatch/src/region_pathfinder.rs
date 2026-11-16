// SPDX-License-Identifier: MIT OR Apache-2.0
//! The multi-node placement pathfinder (ROADMAP row 157, GAP-385).
//!
//! [`PlacementRegionPathfinder`] records one `Op::Branch` per **region**: a single-entry, single-exit
//! stretch of nodes between a diverge node `D` and a reconverge node `R`, where arm 0 is the stretch
//! as placed by the DP winner and arm 1 is a clone of the whole stretch on one alternative placement.
//! The stretch is any connected set of nodes (a chain, a residual block, a fan-out that rejoins), so
//! the picker can choose "stay on this device for these N ops" rather than a one-node swap.
//!
//! Design: `docs/design-notes/2026-11-15-multi-node-placement-pathfinder.md`. In short:
//!
//! 1. Post-dominators over the realize-reachable DAG give, for each node `D`, the candidate reconverge
//!    nodes `R` (its post-dominator chain).
//! 2. The interior of `(D, R)` is every node on a `D -> R` path. A region is valid when `R` reads
//!    exactly one interior node (the exit), no interior node is read from outside (only the exit is read,
//!    and only by `R`), every interior node has a candidate on one common alternative placement that
//!    differs from its winner, and the alternative stretch is competitive on summed time.
//! 3. Valid regions are chosen largest first; regions must be disjoint and neither endpoint may lie in
//!    another chosen interior (a laminar family is NOT built yet: nested regions are a follow-up).
//!
//! This replaces the one-node `PlacementForkPathfinder` in the default pass registries; a one-node
//! region is the old behaviour, so graphs with isolated forks are unchanged.

use std::collections::{HashMap, HashSet};

use fuel_graph::{Graph, Node, NodeId, Op};
use fuel_ir::Result;
use fuel_ir::probe::BackendId;

use crate::driver::{OptimizationContext, Pathfinder};
use crate::ranker::CostVector;

/// Largest interior (in nodes) a region may have. A first-slice cap; the design note requires it be
/// measured on real models before it is fixed.
pub const MAX_REGION_NODES: usize = 64;

/// A region is recorded only when the alternative stretch's summed time is at most this multiple of
/// the winner's. A first-slice bound, to be measured against the per-device frontier.
pub const REGION_COMPETITIVE_FACTOR: u64 = 2;

/// Records multi-node placement forks as `Op::Branch` regions. See the module docs.
pub struct PlacementRegionPathfinder;

struct Region {
    diverge: NodeId,
    reconverge: NodeId,
    /// Interior nodes in dispatch (topological) order.
    interior: Vec<NodeId>,
    exit: NodeId,
    alt_backend: BackendId,
    diverge_idx: usize,
    reconverge_idx: usize,
}

impl Pathfinder for PlacementRegionPathfinder {
    fn name(&self) -> &'static str {
        "PlacementRegionSeed"
    }

    fn propose(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()> {
        let regions = choose_regions(find_regions(graph, ctx));
        for region in regions {
            record_branch(graph, &region);
        }
        Ok(())
    }
}

/// Distinct `(backend, device)` placements of a node's alternative set, in rank order.
fn placements(
    ctx: &OptimizationContext<'_>,
    id: NodeId,
) -> Vec<(BackendId, fuel_ir::DeviceLocation)> {
    let mut out: Vec<(BackendId, fuel_ir::DeviceLocation)> = Vec::new();
    if let Some(set) = ctx.plan.alternatives(id) {
        for c in set.alternatives() {
            let p = (c.backend, c.device);
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Time of the best candidate of `id` on `place`, or `None` when it has none.
fn time_on(
    ctx: &OptimizationContext<'_>,
    id: NodeId,
    place: (BackendId, fuel_ir::DeviceLocation),
) -> Option<u64> {
    let set = ctx.plan.alternatives(id)?;
    set.alternatives()
        .iter()
        .filter(|c| (c.backend, c.device) == place)
        .map(|c| CostVector::from_candidate(c).time)
        .min()
}

fn find_regions(graph: &Graph, ctx: &OptimizationContext<'_>) -> Vec<Region> {
    let order = ctx.order;
    let n = order.len();
    let idx: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Consumers over the realize-reachable set.
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, &id) in order.iter().enumerate() {
        for &input in &graph.node(id).inputs {
            if let Some(&j) = idx.get(&input)
                && !consumers[j].contains(&i)
            {
                consumers[j].push(i);
            }
        }
    }

    // Tensors a destructive op consumes: cloning a reader of one would escape the ordering edge that
    // pins destructive writers after every reader, so regions that read them are skipped.
    let mut destroyed: HashSet<NodeId> = HashSet::new();
    for &id in order {
        let node = graph.node(id);
        if let Some(k) = node.op.destructive_input()
            && let Some(&t) = node.inputs.get(k)
        {
            destroyed.insert(t);
        }
    }

    // Per-node placements and eligibility.
    let places: Vec<Vec<(BackendId, fuel_ir::DeviceLocation)>> =
        order.iter().map(|&id| placements(ctx, id)).collect();
    let eligible: Vec<bool> = order
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let node = graph.node(id);
            places[i].len() >= 2
                && !ctx.cycle_guard.contains(&id)
                && !matches!(node.op, Op::Branch { .. })
                && node.op.destructive_input().is_none()
                && !node.inputs.iter().any(|t| destroyed.contains(t))
        })
        .collect();

    // Immediate post-dominators; index `n` is the virtual exit that every realize root flows to.
    let mut ipdom = vec![n; n + 1];
    for i in (0..n).rev() {
        let cs = &consumers[i];
        if cs.is_empty() {
            ipdom[i] = n;
            continue;
        }
        let mut r = cs[0];
        for &c in &cs[1..] {
            r = intersect(r, c, &ipdom);
        }
        ipdom[i] = r;
    }

    let mut regions = Vec::new();
    for d in 0..n {
        // A region's first node must be a consumer of D and eligible.
        if !consumers[d].iter().any(|&c| eligible[c]) {
            continue;
        }
        let mut r = ipdom[d];
        while r < n {
            let Some(interior) = interior_of(graph, order, &idx, &consumers, d, r) else {
                break; // grew past the cap; every farther R only grows the interior
            };
            if let Some(region) = check_region(
                graph, ctx, order, &idx, &consumers, &places, &eligible, d, r, interior,
            ) {
                regions.push(region);
            }
            r = ipdom[r];
        }
    }
    regions
}

fn intersect(mut a: usize, mut b: usize, ipdom: &[usize]) -> usize {
    while a != b {
        while a < b {
            a = ipdom[a];
        }
        while b < a {
            b = ipdom[b];
        }
    }
    a
}

/// Nodes on a `d -> r` path, strictly between them, in topological order. `None` when the stretch is
/// larger than the cap.
fn interior_of(
    graph: &Graph,
    order: &[NodeId],
    idx: &HashMap<NodeId, usize>,
    consumers: &[Vec<usize>],
    d: usize,
    r: usize,
) -> Option<Vec<usize>> {
    let mut fwd: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = consumers[d].iter().copied().filter(|&c| c < r).collect();
    while let Some(x) = stack.pop() {
        if !fwd.insert(x) {
            continue;
        }
        if fwd.len() > 4 * MAX_REGION_NODES {
            return None;
        }
        stack.extend(consumers[x].iter().copied().filter(|&c| c < r));
    }
    let mut interior: HashSet<usize> = HashSet::new();
    let mut stack: Vec<usize> = graph
        .node(order[r])
        .inputs
        .iter()
        .filter_map(|i| idx.get(i).copied())
        .filter(|x| fwd.contains(x))
        .collect();
    while let Some(x) = stack.pop() {
        if !interior.insert(x) {
            continue;
        }
        stack.extend(
            graph
                .node(order[x])
                .inputs
                .iter()
                .filter_map(|i| idx.get(i).copied())
                .filter(|y| fwd.contains(y)),
        );
    }
    if interior.len() > MAX_REGION_NODES {
        return None;
    }
    let mut v: Vec<usize> = interior.into_iter().collect();
    v.sort_unstable();
    Some(v)
}

#[allow(clippy::too_many_arguments)]
fn check_region(
    graph: &Graph,
    ctx: &OptimizationContext<'_>,
    order: &[NodeId],
    idx: &HashMap<NodeId, usize>,
    consumers: &[Vec<usize>],
    places: &[Vec<(BackendId, fuel_ir::DeviceLocation)>],
    eligible: &[bool],
    d: usize,
    r: usize,
    interior: Vec<usize>,
) -> Option<Region> {
    if interior.is_empty() || !interior.iter().all(|&x| eligible[x]) {
        return None;
    }
    let inside: HashSet<usize> = interior.iter().copied().collect();

    // R reads exactly one interior node: the arm exit.
    let reads: HashSet<usize> = graph
        .node(order[r])
        .inputs
        .iter()
        .filter_map(|i| idx.get(i).copied())
        .filter(|x| inside.contains(x))
        .collect();
    if reads.len() != 1 {
        return None;
    }
    let exit = *reads.iter().next()?;

    // Nothing inside is read from outside: non-exit nodes only by interior nodes, the exit only by R.
    for &x in &interior {
        let cs = &consumers[x];
        let ok = if x == exit {
            cs.len() == 1 && cs[0] == r
        } else {
            cs.iter().all(|c| inside.contains(c))
        };
        if !ok {
            return None;
        }
    }

    // One alternative placement shared by every interior node and different from each node's winner.
    let winners: Vec<(BackendId, fuel_ir::DeviceLocation)> = interior
        .iter()
        .map(|&x| {
            ctx.plan
                .alternatives(order[x])
                .and_then(|s| s.winner())
                .map(|w| (w.backend, w.device))
        })
        .collect::<Option<Vec<_>>>()?;
    let alt = places[interior[0]].iter().copied().find(|p| {
        interior
            .iter()
            .zip(&winners)
            .all(|(&x, w)| p != w && places[x].contains(p))
    })?;

    // Competitive on summed time.
    let mut win_total: u64 = 0;
    let mut alt_total: u64 = 0;
    for (&x, w) in interior.iter().zip(&winners) {
        win_total = win_total.saturating_add(time_on(ctx, order[x], *w)?);
        alt_total = alt_total.saturating_add(time_on(ctx, order[x], alt)?);
    }
    if alt_total > win_total.saturating_mul(REGION_COMPETITIVE_FACTOR) {
        return None;
    }

    Some(Region {
        diverge: order[d],
        reconverge: order[r],
        interior: interior.iter().map(|&x| order[x]).collect(),
        exit: order[exit],
        alt_backend: alt.0,
        diverge_idx: d,
        reconverge_idx: r,
    })
}

/// Largest regions first; disjoint interiors; an endpoint may not lie inside another chosen interior.
fn choose_regions(mut candidates: Vec<Region>) -> Vec<Region> {
    candidates.sort_by_key(|r| {
        (
            std::cmp::Reverse(r.interior.len()),
            r.diverge_idx,
            r.reconverge_idx,
        )
    });
    let mut taken: HashSet<NodeId> = HashSet::new();
    let mut ends: HashSet<NodeId> = HashSet::new();
    let mut chosen = Vec::new();
    for r in candidates {
        let clash = r
            .interior
            .iter()
            .any(|x| taken.contains(x) || ends.contains(x))
            || taken.contains(&r.diverge)
            || taken.contains(&r.reconverge);
        if clash {
            continue;
        }
        taken.extend(r.interior.iter().copied());
        ends.insert(r.diverge);
        ends.insert(r.reconverge);
        chosen.push(r);
    }
    chosen
}

/// Clone the interior on the alternative placement and record the 2-arm branch. A rejected branch is
/// not an error (as for the one-node fork): the clones are unreachable and the single route is kept.
fn record_branch(graph: &mut Graph, region: &Region) {
    let mut clone_of: HashMap<NodeId, NodeId> = HashMap::new();
    for &x in &region.interior {
        let (op, inputs, shape, dtype) = {
            let n = graph.node(x);
            (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
        };
        let inputs = inputs
            .into_iter()
            .map(|i| clone_of.get(&i).copied().unwrap_or(i))
            .collect();
        let c = graph.push(Node {
            op,
            inputs,
            shape,
            dtype,
        });
        graph.set_target_backend(c, region.alt_backend);
        clone_of.insert(x, c);
    }
    let arm1 = clone_of[&region.exit];
    let mut builder = graph.open_branch(region.diverge);
    builder.add_arm(region.exit); // arm 0 = the DP winner's stretch, in place
    builder.add_arm(arm1);
    let _ = builder.finalize_branches(graph, region.reconverge);
}
