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

/// Records multi-node placement forks as `Op::Branch` regions. See the module docs.
pub struct PlacementRegionPathfinder;

#[derive(Clone)]
struct Region {
    diverge: NodeId,
    reconverge: NodeId,
    /// Interior nodes in dispatch (topological) order.
    interior: Vec<NodeId>,
    exit: NodeId,
    alt_backend: BackendId,
    /// The device the alternative placement runs on (the plan's own device for it).
    alt_device: fuel_ir::DeviceLocation,
    /// The device arm 0's exit lives on (its DP winner's), which the reconverge node reads.
    exit_device: fuel_ir::DeviceLocation,
    diverge_idx: usize,
    reconverge_idx: usize,
}

impl Pathfinder for PlacementRegionPathfinder {
    fn name(&self) -> &'static str {
        "PlacementRegionSeed"
    }

    fn propose(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()> {
        if std::env::var_os("FUEL_PRINT_BRANCH_COUNT").is_some() {
            let mut hist = [0usize; 5];
            for &id in ctx.order {
                hist[placements(ctx, id).len().min(4)] += 1;
            }
            eprintln!(
                "fuel: region pathfinder sees {} nodes; placements-per-node histogram [0,1,2,3,4+] = {:?}; pinned/cycle-guarded = {}",
                ctx.order.len(),
                hist,
                ctx.cycle_guard.len()
            );
        }
        let found = find_regions(graph, ctx);
        if std::env::var_os("FUEL_PRINT_BRANCH_COUNT").is_some() {
            eprintln!(
                "fuel: region pathfinder found {} candidate region(s)",
                found.len()
            );
        }
        let candidates = found.len();
        if std::env::var_os("FUEL_PRINT_BRANCH_COUNT").is_some() {
            let c = policy_counts(&found);
            eprintln!(
                "fuel: region policy counts over {candidates} candidate(s) (diagnostic only; today's choice is the first): \
                 today (largest first, disjoint) = {}, laminar family (nested allowed; NOT buildable yet) = {}, \
                 smallest first, disjoint = {}, one-node regions only, disjoint = {}",
                c.today, c.laminar, c.smallest_first, c.one_node
            );
        }
        let regions = choose_regions(found);
        let chosen = regions.len();
        let (mut recorded, mut collapsed) = (0usize, 0usize);
        let mut rejected: Vec<String> = Vec::new();
        for region in regions {
            // A rejected or collapsed branch is not an error (the single route is kept); the outcome is
            // only counted, for the `FUEL_PRINT_BRANCH_COUNT` report below.
            match try_record_branch(
                graph,
                region.diverge,
                region.reconverge,
                &region.interior,
                region.exit,
                region.alt_backend,
                region.alt_device,
                region.exit_device,
            ) {
                Ok(Some(_)) => recorded += 1,
                Ok(None) => collapsed += 1,
                Err(e) => rejected.push(format!("{e:?}")),
            }
        }
        crate::optimizer_trace::global().record_regions(
            candidates - chosen,
            recorded,
            collapsed,
            &rejected,
        );
        if std::env::var_os("FUEL_PRINT_BRANCH_COUNT").is_some() {
            eprintln!(
                "fuel: region pathfinder: {candidates} candidate(s) -> {chosen} chosen by choose_regions \
                 ({} dropped as overlapping an already-chosen larger region) -> {recorded} recorded, \
                 {collapsed} collapsed, {} rejected by finalize_branches",
                candidates - chosen,
                rejected.len()
            );
            for reason in rejected.iter().take(5) {
                eprintln!("fuel: region pathfinder: rejected: {reason}");
            }
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
    // The DP prunes each node's set to its committed device, so a second placement lives in the plan's
    // off-device side map (read here; the realize path never reads it).
    for set in [
        ctx.plan.alternatives(id),
        ctx.plan.off_device_alternatives(id),
    ]
    .into_iter()
    .flatten()
    {
        for c in set.alternatives() {
            let p = (c.backend, c.device);
            if !out.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// `(compute_ns, inbound_transfer_ns)` of the best candidate of `id` on `place` (best by their sum),
/// or `None` when it has none. A candidate's `time` includes the transfer of its inputs onto its
/// device, which is only paid where a value actually crosses the region boundary, so the two terms are
/// kept apart.
fn time_on(
    ctx: &OptimizationContext<'_>,
    id: NodeId,
    place: (BackendId, fuel_ir::DeviceLocation),
) -> Option<(u64, u64)> {
    [
        ctx.plan.alternatives(id),
        ctx.plan.off_device_alternatives(id),
    ]
    .into_iter()
    .flatten()
    .flat_map(|set| set.alternatives().iter())
    .filter(|c| (c.backend, c.device) == place)
    .map(|c| {
        let total = CostVector::from_candidate(c).time;
        (
            total.saturating_sub(c.inbound_transfer_ns),
            c.inbound_transfer_ns,
        )
    })
    .min_by_key(|&(compute, inbound)| compute.saturating_add(inbound))
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
        // Only an entry node (one reading a value from outside the stretch) pays an inbound transfer;
        // interior edges stay on one device inside an arm. The exit transfer back is not priced here
        // (unmeasured; the DP prices it when it commits a placement).
        let is_entry = graph
            .node(order[x])
            .inputs
            .iter()
            .any(|i| !idx.get(i).is_some_and(|j| inside.contains(j)));
        let (wc, wi) = time_on(ctx, order[x], *w)?;
        let (ac, ai) = time_on(ctx, order[x], alt)?;
        win_total = win_total.saturating_add(wc);
        alt_total = alt_total.saturating_add(ac);
        if is_entry {
            win_total = win_total.saturating_add(wi);
            alt_total = alt_total.saturating_add(ai);
        }
    }
    if std::env::var_os("FUEL_PRINT_BRANCH_COUNT").is_some() {
        eprintln!(
            "fuel: region d={d} r={r} interior={} win_total={win_total} alt_total={alt_total}",
            interior.len()
        );
    }
    // No competitiveness gate: `win_total`/`alt_total` are idle-system estimates, and which arm is faster
    // depends on the device load at run time (a GPU with an hour of queued work makes a short CPU stretch
    // the faster path). Every feasible alternative stays a branch arm and the executor's picker decides.
    // The bound is structural: one alternative per region, interiors disjoint, each at most
    // `MAX_REGION_NODES` (still an unmeasured placeholder), so the arms clone at most the graph's own size.

    Some(Region {
        diverge: order[d],
        reconverge: order[r],
        interior: interior.iter().map(|&x| order[x]).collect(),
        exit: order[exit],
        alt_backend: alt.0,
        alt_device: alt.1,
        exit_device: interior
            .iter()
            .position(|&x| x == exit)
            .and_then(|p| winners.get(p))
            .map(|w| w.1)?,
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
    choose_in_order(candidates)
}

/// How many regions each selection policy would keep from the same candidates. Diagnostic only
/// (`FUEL_PRINT_BRANCH_COUNT`): the pathfinder's behaviour is [`choose_regions`] alone.
struct PolicyCounts {
    today: usize,
    laminar: usize,
    smallest_first: usize,
    one_node: usize,
}

fn policy_counts(found: &[Region]) -> PolicyCounts {
    let today = choose_regions(found.to_vec()).len();

    let mut asc = found.to_vec();
    asc.sort_by_key(|r| (r.interior.len(), r.diverge_idx, r.reconverge_idx));
    let smallest_first = choose_in_order(asc).len();

    let mut one: Vec<Region> = found
        .iter()
        .filter(|r| r.interior.len() == 1)
        .cloned()
        .collect();
    one.sort_by_key(|r| (r.diverge_idx, r.reconverge_idx));
    let one_node = choose_in_order(one).len();

    PolicyCounts {
        today,
        laminar: laminar_count(found),
        smallest_first,
        one_node,
    }
}

/// Greedy largest-first selection that keeps a pair when its interiors are disjoint (with the same
/// endpoint rule as today) or strictly nested with the inner region's endpoints on or inside the outer
/// one, and drops a pair that overlaps without nesting (`docs/design-notes/2026-11-15-multi-node-
/// placement-pathfinder.md`, section 5). Counting only: nested branches are not built.
fn laminar_count(found: &[Region]) -> usize {
    let mut sorted: Vec<&Region> = found.iter().collect();
    sorted.sort_by_key(|r| {
        (
            std::cmp::Reverse(r.interior.len()),
            r.diverge_idx,
            r.reconverge_idx,
        )
    });
    let mut kept: Vec<(&Region, HashSet<NodeId>)> = Vec::new();
    for r in sorted {
        let inner: HashSet<NodeId> = r.interior.iter().copied().collect();
        let ok = kept.iter().all(|(q, outer)| {
            if inner.is_disjoint(outer) {
                !outer.contains(&r.diverge)
                    && !outer.contains(&r.reconverge)
                    && !inner.contains(&q.diverge)
                    && !inner.contains(&q.reconverge)
            } else if inner.is_subset(outer) {
                (outer.contains(&r.diverge) || r.diverge == q.diverge)
                    && (outer.contains(&r.reconverge) || r.reconverge == q.reconverge)
            } else {
                false
            }
        });
        if ok {
            kept.push((r, inner));
        }
    }
    kept.len()
}

/// The disjointness rule of [`choose_regions`], applied in the order given.
fn choose_in_order(candidates: Vec<Region>) -> Vec<Region> {
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

/// Clone the interior on the alternative placement and record the 2-arm branch, returning the branch
/// node when it was recorded. A rejected branch is not an error (as for the one-node fork): the clones
/// are unreachable and the single route is kept. `pub(crate)` so tests can build the exact structure
/// the pathfinder builds with an alternative backend that can run on the host.
#[cfg(test)]
pub(crate) fn record_branch(
    graph: &mut Graph,
    diverge: NodeId,
    reconverge: NodeId,
    interior: &[NodeId],
    exit: NodeId,
    alt_backend: BackendId,
) -> Option<NodeId> {
    // A host test double: the alternative runs on the backend's default device and arm 0's exit lives
    // there too, so no exit bridge is added (a test that wants one calls `try_record_branch`).
    let device = alt_backend.default_location();
    try_record_branch(
        graph,
        diverge,
        reconverge,
        interior,
        exit,
        alt_backend,
        device,
        device,
    )
    .ok()
    .flatten()
}

/// [`record_branch`] with the outcome kept: `Ok(Some(branch))` recorded, `Ok(None)` collapsed to one arm,
/// `Err` rejected by `finalize_branches` (the reason names the rule).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_record_branch(
    graph: &mut Graph,
    diverge: NodeId,
    reconverge: NodeId,
    interior: &[NodeId],
    exit: NodeId,
    alt_backend: BackendId,
    alt_device: fuel_ir::DeviceLocation,
    exit_device: fuel_ir::DeviceLocation,
) -> Result<Option<NodeId>> {
    let mut clone_of: HashMap<NodeId, NodeId> = HashMap::new();
    for &x in interior {
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
        graph.set_target_backend(c, alt_backend);
        // The clone's PLACEMENT, not only its backend stamp: the residency pass places a node with no
        // placement and no plan entry on the pinned realize device, so without this it never sees the
        // arm's inputs cross a device and inserts no inbound copy (ROADMAP row 158: a CPU arm in a Vulkan
        // realize read Vulkan storage directly, "cpu kernel wrapper called with non-CPU input").
        graph.set_placement(c, alt_device);
        clone_of.insert(x, c);
    }
    let mut arm1 = clone_of[&exit];
    // The reconverge node reads ARM 0's exit, which lives on `exit_device`, and the executor aliases the
    // chosen arm's exit onto it with a zero-copy view. When the alternative runs elsewhere, bring the
    // arm's result back with an explicit copy so both exits live on the same device.
    if alt_device != exit_device {
        let (shape, dtype) = {
            let n = graph.node(arm1);
            (n.shape.clone(), n.dtype)
        };
        let back = graph.push(Node {
            op: Op::Copy {
                target: exit_device,
            },
            inputs: vec![arm1],
            shape,
            dtype,
        });
        // A copy runs where its bytes come from (the residency pass re-stamps it the same way).
        graph.set_target_backend(back, alt_backend);
        graph.set_placement(back, exit_device);
        arm1 = back;
    }
    let mut builder = graph.open_branch(diverge);
    builder.add_arm(exit); // arm 0 = the DP winner's stretch, in place
    builder.add_arm(arm1);
    builder.finalize_branches(graph, reconverge)
}

#[cfg(test)]
mod choose_regions_tests {
    //! Pins TODAY's `choose_regions` policy (ROADMAP row 157): largest interior first, interiors
    //! disjoint, no chosen region's endpoint inside another chosen interior, one survivor per
    //! overlapping family. One test per drop reason, plus controls. They pin behaviour; they do not
    //! argue it is the right policy (a laminar family would keep nested regions, a separate row).
    use super::*;

    fn region(d: usize, r: usize, interior: &[usize]) -> Region {
        let ids: Vec<NodeId> = interior.iter().map(|&x| NodeId(x)).collect();
        Region {
            diverge: NodeId(d),
            reconverge: NodeId(r),
            exit: *ids.last().expect("non-empty interior"),
            interior: ids,
            alt_backend: BackendId::Cuda,
            alt_device: fuel_ir::DeviceLocation::Cuda { gpu_id: 0 },
            exit_device: fuel_ir::DeviceLocation::Cpu,
            diverge_idx: d,
            reconverge_idx: r,
        }
    }

    /// `(diverge, reconverge)` of every chosen region, in choice order.
    fn chosen(candidates: Vec<Region>) -> Vec<(usize, usize)> {
        choose_regions(candidates)
            .iter()
            .map(|r| (r.diverge.0, r.reconverge.0))
            .collect()
    }

    /// Region A: 0 -> 10 over interior {1,2,3}. The larger region in every test below.
    fn a() -> Region {
        region(0, 10, &[1, 2, 3])
    }

    #[test]
    fn control_disjoint_regions_are_all_kept() {
        let b = region(20, 30, &[21, 22]);
        assert_eq!(chosen(vec![b, a()]), vec![(0, 10), (20, 30)]);
    }

    #[test]
    fn control_regions_chained_at_an_endpoint_are_both_kept() {
        // B starts at A's reconverge node (10): sharing an endpoint is allowed, no interior node is shared.
        let b = region(10, 20, &[11, 12]);
        assert_eq!(chosen(vec![b, a()]), vec![(0, 10), (10, 20)]);
    }

    #[test]
    fn drop_reason_1_interior_overlaps_a_chosen_interior() {
        let b = region(20, 30, &[3, 4]); // shares node 3 with A's interior
        assert_eq!(chosen(vec![b, a()]), vec![(0, 10)]);
    }

    #[test]
    fn drop_reason_2_interior_contains_a_chosen_endpoint() {
        let b = region(8, 12, &[10, 11]); // node 10 is A's reconverge
        assert_eq!(chosen(vec![b, a()]), vec![(0, 10)]);
    }

    #[test]
    fn drop_reason_3_diverge_lies_in_a_chosen_interior() {
        let b = region(2, 20, &[40, 41]); // diverge 2 is inside A's interior
        assert_eq!(chosen(vec![b, a()]), vec![(0, 10)]);
    }

    #[test]
    fn drop_reason_4_reconverge_lies_in_a_chosen_interior() {
        let b = region(30, 3, &[50, 51]); // reconverge 3 is inside A's interior
        assert_eq!(chosen(vec![b, a()]), vec![(0, 10)]);
    }

    #[test]
    fn the_larger_of_two_overlapping_regions_wins_regardless_of_input_order() {
        // Nested family: same diverge, growing reconverge (what the toy Llama log shows: d=7, r=156..159).
        let small = region(7, 20, &[8]);
        let large = region(7, 22, &[8, 20, 21]);
        assert_eq!(chosen(vec![small, large]), vec![(7, 22)]);
        let small = region(7, 20, &[8]);
        let large = region(7, 22, &[8, 20, 21]);
        assert_eq!(chosen(vec![large, small]), vec![(7, 22)]);
    }

    #[test]
    fn policy_counts_on_a_nested_family_and_on_a_partial_overlap() {
        // The diagnostic that feeds the policy decision must itself be checked on known shapes.
        let nested = vec![
            region(7, 20, &[8]),
            region(7, 21, &[8, 20]),
            region(7, 22, &[8, 20, 21]),
            region(7, 23, &[8, 20, 21, 22]),
        ];
        let c = policy_counts(&nested);
        assert_eq!(
            (c.today, c.laminar, c.smallest_first, c.one_node),
            (1, 4, 1, 1),
            "nested family: today keeps the largest only; a laminar family keeps all four",
        );
        // Overlap WITHOUT nesting (shares node 3, neither contains the other): no policy keeps both.
        let partial = vec![region(0, 10, &[1, 2, 3]), region(5, 15, &[3, 4, 6])];
        let c = policy_counts(&partial);
        assert_eq!(
            (c.today, c.laminar, c.smallest_first, c.one_node),
            (1, 1, 1, 0),
            "partial overlap is dropped by every policy; no interior has one node",
        );
        // Disjoint regions: every policy keeps both (one-node policy only sees size-1 interiors).
        let disjoint = vec![region(0, 10, &[1]), region(20, 30, &[21])];
        let c = policy_counts(&disjoint);
        assert_eq!(
            (c.today, c.laminar, c.smallest_first, c.one_node),
            (2, 2, 2, 2)
        );
    }

    #[test]
    fn a_family_of_four_nested_candidates_leaves_one_survivor() {
        // Interiors of size 1..=4, same diverge: the shape behind "100 candidates -> 18 chosen".
        let family = vec![
            region(7, 20, &[8]),
            region(7, 21, &[8, 20]),
            region(7, 22, &[8, 20, 21]),
            region(7, 23, &[8, 20, 21, 22]),
        ];
        assert_eq!(chosen(family), vec![(7, 23)]);
    }
}
