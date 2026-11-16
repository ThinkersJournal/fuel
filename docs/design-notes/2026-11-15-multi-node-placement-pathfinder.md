# Multi-node placement pathfinder (ROADMAP row 157, GAP-385)

Status: DESIGN, not implemented. Facts below are read at `origin/main` `ad9466459` unless marked
**hypothesis**. Author: Fuel lane. Reviewer: PM, CireSnave.

## 1. What exists

- `Graph::open_branch(diverge)`, `BranchBuilder::add_arm(exit)`, `finalize_branches(reconverge_at)`
  (`fuel-graph/src/lib.rs`) build an `Op::Branch { reconverge_at }` node whose inputs are the arm exits.
  Validation (all before any mutation): `reconverge_at` is a strict forward descendant of `diverge`;
  arms are internally disjoint (interior nodes, i.e. nodes on paths `diverge -> exit`, are neither
  shared between arms nor read by anything outside the branch, except `reconverge_at` reading an exit);
  all exits agree on shape and dtype; `reconverge_at` reads arm 0's exit, so an unpicked graph still
  realizes. A single-arm branch collapses to that arm.
- `PlacementForkPathfinder` (`fuel-dispatch/src/driver.rs`) is the only placement pathfinder. It records a
  branch for a node with >=2 distinct `(backend, device)` placements in the plan's alternative set and
  exactly one consumer; arm 0 is the node, arm 1 a clone on the runner-up placement, reconverge is the sole
  consumer. Every branch it makes is one node wide.
- The placement DP is a carry-forward `(position x device)` search with transfer pricing
  (`04-optimization.md`, load-time planning). It reasons over sequences; the pathfinder exposes only its
  single-node leftovers.

## 2. Target

A decision point is a `(diverge D, reconverge R)` pair with >=2 arms, each arm a connected stretch of
nodes between them, differing in placement (and later kernel variant, fusion, algebraic form, tolerance).
The picker resolves arms at runtime exactly as today (`RuntimeSelector`); only the supply of branches
changes.

## 3. Choosing regions

A candidate region is single-entry, single-exit with respect to `(D, R)`: every path from D to R lies
inside the region, and nothing inside it is read from outside except through R.

1. Compute dominators and post-dominators over the realize-reachable DAG.
2. For each node D, the candidate Rs are its post-dominators. The region is every node on a D -> R path.
3. Score each region with the DP's carry-forward cost for the winning placement sequence and for the best
   sequence that differs on at least one node of the region (the runner-up *stretch*, not the runner-up
   *node*).
4. Keep a region only if the runner-up stretch is within the frontier's competitiveness bound (section 7).

**Fan-out.** A node x with several consumers is an ordinary interior node. The region simply extends to the
post-dominator that joins all of x's consumers, so no tensor read outside the arm is interior to it.
x as the *diverge* point (its value feeds both arms) is also fine: arms may read upstream of D freely.

**Residual join.** `y = x + f(x)`: D = x, R = the add. Arm interior = `f`'s nodes. The edge `x -> add` bypasses
the arm and reads D directly, which is allowed (D is shared, not interior). The block is one decision
point with D = block input and R = the residual add.

## 4. Building arms

- Arm 0 is the winner's nodes in place (they already exist; nothing is cloned for it).
- Arm k >= 1 is a clone of the region's nodes with `target_backend` set to the arm's placement, reading
  the same upstream inputs, orphaned (read only by the `Branch`) so live data flow is unchanged.
- Placement changes at the arm's entry or exit insert transfers (`Op::Move { target }`), counted in the
  arm cost. The exit must have the shape and dtype of arm 0's exit; a differing dtype is cast *inside* the
  arm (the builder validates, it does not insert casts).

## 5. When arms disagree (fallback rules)

| Disagreement | Rule |
| --- | --- |
| Exit shape or dtype differs | Cast inside the arm before the exit; if impossible, drop the region (no branch). |
| Arm 0 not runnable | Not recorded: `finalize_branches` already refuses; realize proceeds on the unchanged single route. |
| Runner-up kernel missing, device absent, or over the memory budget | Region not recorded. |
| Numerics differ (kernel accuracy claims differ) | The existing precision filter and `AccuracyPreferenceFilter` decide which arms are admissible; arm 0 is the admissible winner. |
| Costs near-tie | Keep both arms; the selector chooses by telemetry. |
| Two regions overlap without nesting | Regions must form a laminar family (nested or disjoint). Overlapping non-nested regions merge into their union or the lower-scoring one is dropped. |
| `finalize_branches` rejects a candidate | Non-fatal, as today: the region is not recorded, no clone is reachable. |

## 6. Coupling

Nested branches make the selector's choice at an inner point depend on the outer arm (an outer arm on the
CPU makes an inner GPU arm moot). The existing per-decision-point coupling adjustments
(`04-optimization.md`, "What counts as a decision point") are the mechanism; **hypothesis:** they already
handle nesting. Verify with a red test before relying on it.

## 7. Bounding

No fixed top-N. The bound is the per-device Pareto frontier plus its crowding-distance cap (the existing
constitution bound), plus two new caps that must be measured before they are fixed: maximum region length
and maximum branches per realize. The one-consumer "fewness gate" is retired; the caps replace it.

## 8. Tests (red first)

Fixtures reuse `two_backend_opts` and the branch graph builders in `fuel-dispatch/src/optimize.rs` tests.

1. **Chain stretch (born red):** `a -> b -> c -> d -> e` with both backends able to run b, c, d. Expect one
   `Op::Branch` with D = a, R = e, arms of length 3. Today's pathfinder yields one-node forks or none, so
   this is red for the right reason.
2. **Fan-out:** `x -> {p, q} -> join`, p and q each two ops. Expect a decision point whose region covers both
   consumers; no interior node is read from outside.
3. **Residual join:** `y = x + f(x)` with `f` three ops. Expect D = x, R = the add, arm interior = f.
4. **Disagreement:** an arm whose exit dtype differs is cast inside the arm; an arm with a missing kernel
   is not recorded; arm 0 is always runnable.
5. **Sabotage siblings:** each test has a variant that violates one validation rule (shared interior node,
   interior node read outside) and must produce no branch, not a panic.
6. **No-op check:** a graph with a single viable placement yields zero branches (the fewness property the
   old gate protected, now by competitiveness).

## 9. Not verified

- How many alternatives per node the DP and ranker keep today; whether a clearly slower CPU placement
  survives pruning (decides how many regions exist on a real model).
- Whether `RuntimeSelector` resolves nested branches (section 6).
- Cost of cloned arms on a real model graph (node and memory growth).
- Whether a diverge node with multiple inputs inside the stretch needs anything beyond "arms read upstream
  of D"; the builder rule reads that way but the red tests are the proof.

## 10. Order of work

1. This note reviewed. 2. Red tests 1 to 3. 3. Region finder (dominators, SESE regions, stretch scoring).
4. Arm builder with transfers. 5. Disagreement rules and laminar check. 6. Measure on a real model graph and
fix the caps. 7. Later pathfinders for kernel-variant, fusion-vs-decomposition, algebraic, tolerance
alternatives.
