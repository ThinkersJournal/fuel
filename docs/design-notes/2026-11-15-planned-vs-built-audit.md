# Planned vs built: audit of the architecture docs against the code (PM task 2026-11-15)

Ref: code at `origin/main` `ad9466459` (the audited worktree differs only by docs). Method: four read-only
passes (IR+persistence, optimization+tolerance+harvest, runtime+lifecycle+consumer contract, identity+backend
contract+layers+non-goals) with `git grep` over `*.rs`, in-file `#[cfg(test)]` modules and `tests/` excluded
unless stated. **No test was run**: every "test" below is a name found by grep. Counts of 0 are paired with a
positive control only where stated. The lane spot-checked 16 headline claims itself (marked **V**); the rest are
the passes' findings, labelled as such. "Doc" = `docs/architecture/<file>`.

Status words: BUILT = on the production realize path. PARTIAL = exists, limited (how is stated). STUB = a type
exists, nothing constructs or reads it. ABSENT = 0 hits.

## 1. The multi-path DAG (edge 3, `03-ir` "optimized form", `04` per-decision-point alternatives)

| Component | Status | Predicate and ref |
| --- | --- | --- |
| `Op::Branch` + builders (`open_branch`, `add_arm`, `finalize_branches`) | BUILT, general | `fuel-graph/src/lib.rs`: any diverge node, any strict-descendant reconverge, multi-node arms legal (rules: disjoint interiors, uniform exit shape/dtype, arm 0 readable) |
| Executor consumes branches | BUILT | `resolve_branch` at `pipelined.rs:2520,2756` via `walk_picked_route_streaming`; `streaming_pick_for` returns None when no `Op::Branch` exists |
| Who CONSTRUCTS branches on the realize path | PARTIAL: 3 sites | `git grep open_branch(` non-test: `driver.rs:388` PlacementForkPathfinder; `decode_flash.rs:276`; `runtime_fused_arm.rs:105`. **V** |
| Placement fork | PARTIAL: one node, 2 arms, exactly one consumer | `driver.rs:291`: arm 1 is an orphan clone of the one node; fan-out and residual joins never fork; `finalize_branches` rejection ignored (`let _ =`). **V** (read in full) |
| Decode flash arm | PARTIAL: CUDA-only, llama-family decode, **baked out before runtime** | `bake_variant_branches` collapses same-device branches (`optimize.rs` ~405); not preserved for adaptation |
| Runtime-fused arm | PARTIAL: wired, no production feeder | `adopt_runtime_fused` non-test callers only in `jit_*` and CUDA live tests; `jit` feature off by default; arm pinned to Cpu |
| Live branches on a real model | NOT MEASURED | `branch_count(` has 0 callers outside `optimize.rs` unit tests (**V**). Placement forks need >=2 admitted placements; CPU-only or single device = 0 branches |
| Frontier over whole paths | PARTIAL: per node | `retain_per_device_frontier` (`alternative_set.rs:255`, `KEEP_PER_DEVICE = 32`, a const, not a tunable); only winner + first runner-up reach the graph |
| Kernel-variant alternatives as decision points | PARTIAL | enumerated into a transient per-node set; only `winner.backend` is stamped; executor takes the first non-broadcast binding (`compiled.rs:301`) |
| Fusion-vs-decomposition, algebraic, tolerance alternatives | ABSENT as producers | only 2 `impl Pathfinder` exist (`PlacementForkPathfinder`, `RuntimeFusedArmPathfinder`) |
| Coupling between decisions | STUB | `.coupling` read in 1 place, a test asserting it is empty (**V**) |
| Wall-clock cost (max over parallel arms) | ABSENT | cost is per node, serial |

## 2. Base map, rewrite engine, tolerance, harvest

| Component | Status | Predicate and ref |
| --- | --- | --- |
| Base map permanently retained; re-optimization restarts from it | ABSENT | `lower_to_base_map(` callers outside `opt.rs`: `jit_ingest.rs`, `jit_ingest_probe.rs`, `runtime_fused.rs`, a `scan.rs` test (hashing only) (**V**); `optimize_graph` mutates the one graph |
| `RuleRegistry` / `FusionRule` / find-best-cover on realize | ABSENT on realize | `optimize_to_fixpoint`, `default_rules(`, `capability_gated_rules(` non-test callers: 0 (hits are a doc comment and test modules in `lazy.rs`, `awq.rs`) (**V**) |
| Algebraic rewrites (CSE, folds, reassociation) | PARTIAL: test-only | `opt::optimize` 1 non-test caller (`runtime_fused.rs`); distributivity/reassociation: none |
| `decompose` on the realize path | PARTIAL | callers: backward pre-pass, cost (unused), JIT, and `PAGED_ATTN` lowering on CUDA only; totality over all fused ops not verified |
| Fused-op cost from decompose (Layer-1 default) | STUB | `fused_layer1_cost`/`cost_from_decompose` callers outside `fused_cost.rs`: 0 (**V**) |
| Per-op/per-region tolerance budgets, cumulative error, per-call override (07) | ABSENT | `ToleranceBudget\|ErrorBound\|tolerance_budget` in 0 `.rs` files (**V**); only a global `PrecisionRequirement` + `AccuracyPreferenceFilter` |
| Pattern harvest (08) | ABSENT | 0 hits for harvest types; the telemetry feed is a different Baracuda flow (feature `telemetry`, default off) |
| Rules declare cost/error/frontier-compat; `Concurrent`/`WholeGraph` | ABSENT | `Rule` trait has name/family/matches/rewrite only; `WholeGraph\|FrontierCompat\|enum Concurrency` 0 hits |
| Rankers/optimizers as pass kinds | PARTIAL | no `Ranker` trait; ranking inline in `compile_plan`; `FrontierConvergenceOptimizer` only `debug_assert!`s |
| Hardcoded optimizer passes outside the rule registry | VIOLATES 09 | `optimize.rs` 289-460: dtype fixups, residency copies, layout fixups, variant bake, PAGED_ATTN lowering |

## 3. Optimization timing and transactions (rows 88, 89)

| Component | Status | Predicate and ref |
| --- | --- | --- |
| Concurrent optimize-and-execute, frontier, commit, regional swap | ABSENT | `GraphTransaction\|OptimizationFrontier\|advance_frontier\|ArcSwap\|DependencyRecord` 0 hits (**V**); optimization runs synchronously under the graph write lock before the first launch (`pipelined_bridge.rs:919` -> `:893`) |
| Planning starts at model load | ABSENT | `fuel-tensor/src/planner.rs:51` `Planner::warm` is `Ok(())` |
| Background / scoped re-optimization | ABSENT | 0 hits for `reoptimi`; `TopologyChanged` retry rebuilds the whole plan |
| Placement carry-forward DP with transfer pricing | BUILT (batch) | `placement_dp.rs` via `plan.rs:708`; one end-of-walk backtrack; no fused-jump lookback; fan-out chains the first consumer only |
| Plan-fragment memoization | ABSENT | `PlanStore` deleted (4 hits, comments) |
| Horizontal fusion; weight-prefetch schedule | ABSENT | 0 hits; `prefetch` in 0 `.rs` files, `madvise\|WILLNEED` 0 (**V**, control `WriteSlice` finds 50 files) |

## 4. Executor and runtime (06, 14, 15)

| Component | Status | Predicate and ref |
| --- | --- | --- |
| Per-branch runtime picker (VRAM, live load, Judge) | BUILT | `production_selector_for` -> `ChainedSelector`; `slot_capacity()` hardwired 1 (`pipelined_bridge.rs:1838`) |
| `pick_route` / `RouteCache` / `LOOKAHEAD_K` | STUB | not called on the realize path (**V**: `pick_route(` in `pipelined.rs` only in comments and one test; `LOOKAHEAD_K` defined and re-exported, unused) |
| Runtime Kahn ready-set scheduler, slot assignment, "runs as a unit" | ABSENT | executor consumes a pre-flattened order, one `WorkItem` per node; Kahn exists only as a static reorder in `fuel-graph/src/run.rs` |
| Lookahead bound | ABSENT | unbounded std channel producer -> executor |
| Memory-pressure serialization; cancellation; serial-reduction ordering for determinism | ABSENT | `CancellationToken\|cancel_token` 0 hits (06 and 15 C-2 contradict each other) |
| Cross-device transfers (Copy/Move) | BUILT | optimizer residency pass; executor waits on copy producers |
| Plan-once decode, paged KV pool | BUILT | `DecodeSession` -> `realize_one_prebuilt_cached_env`; `fuel-kv-pool` |
| Captured-run replay | PARTIAL | CUDA opt-in for llama/phi only; Vulkan not wired |
| Persisted graph, plan cache, fingerprint, snapshots, `.fuel-tolerance` (11) | ABSENT | `map_from_file` 0 `.rs` files (**V**); no graph serializer; only the Judge profile persists |
| Bundled Judge baseline; online Judge accumulation | ABSENT | `populate_dispatch_table` has 0 callers; realize only reads a cached report |
| Consumer contract C-1..C-7 | PARTIAL each | capacity partial, no cancel, state externalization lossy end only, no per-unit cost record, no declared-mutation API, no observe/intervene (`Op::Observe\|Op::Tap` 0 hits) |

## 5. IR leftovers, backend contract, layering

| Component | Status | Predicate and ref |
| --- | --- | --- |
| Storage classes, sessions as IR | STUB / ABSENT | `.storage_class(` 8 callers, all in the `fuel-graph` test module (**V**); no `(NodeId, SessionId)` table |
| `Op::Scan` early-exit driver; Hopfield consumer | STUB / ABSENT | `drive_scan` 0 hits; `hopfield_retrieve` 0 |
| `promote_views_for_liveness` | PARTIAL: tests only | callers in `fuel-dispatch`/`fuel-tensor`: 0 (**V**) |
| Backends advertise capabilities | PARTIAL | dispatch derives them (`derive_backend_caps`); `impl BackendCapabilityProvider for` = 1, a test dummy (**V**) |
| `BackendIdentity`, `BackendPressureSignals`, `BackendDiagnostics` | ABSENT | no such traits (guarded by `doc_declared_trait_allowlist.txt`; `trait BackendIdentity` 1 hit = the guard) |
| Compliance table (05) | WRONG | Caps all checked but 0 impls; Vulkan Streams/Pressure "inherent" not found; Metal row missing |
| Backend crates must not depend on `fuel-dispatch` | VIOLATED by 2 | `fuel-aocl-cpu-backend/Cargo.toml:18`, `fuel-mkl-cpu-backend/Cargo.toml:17` (**V**) |
| No silent fast_math | VIOLATED (Metal) | `fuel-metal-kernels/src/kernel.rs:186` env default `true` (**V**) |
| Always-built per-primitive bit_stable lint | NOT FOUND | only the GAP-077 simulated-flip test |
| `fill_unset_cpu_precision` "retires" | STILL RUNS | `dispatch.rs:7399` (**V**) |
| Crate split in 02 (`fuel-model-*`, `fuel-format-*`) | PARTIAL / stale | only `fuel-model-llama`, `fuel-model-phi` exist; `fuel-tensor`/`fuel-loaders` exist though the diagram marks them absent |

## 6. Places where docs and code disagree (docs to correct; none changes a code claim)

1. `03-ir` 121-135 base map retained / re-optimization; `11-persistence` whole doc; `03-ir` 168-176 persisted graph: not built (sections 2, 4).
2. `03-ir` 137-153, `01-identity` E3, `04` 72-140: frontier of whole paths, kernel/fusion/algebraic/tolerance decision points: section 1.
3. `04` 45 "optimization IS lower-to-base-map-then-cover"; `04` 303 fused cost from decompose; `04` 184-207 planning at load; `04` 255-274 fragments/prefetch; `04` 62-70 rule properties; `04` 132 `keep` knob; `04` 105-125 coupling.
4. `04` 371 says the declarative pattern engine is stubbed; code implements it (`opt.rs` 658-739).
5. `06` 45/51/55/69/79-95/109-172: `pick_route`, `RouteCache`, `LOOKAHEAD_K`, bounded lookahead, Kahn scheduler, memory-pressure serialization, prefetch, background and scoped re-optimization, bundled Judge, mmap'd cache.
6. `14` Stage 5 says production decode does not use plan-once; it does (`DecodeSession`). `14` Stage 4 line citations are stale.
7. `06` "cancellation not supported" vs `15` C-2 "can cancel".
8. `02-layers` rule-1 list names `fuel-nn`/`fuel-transformers` as violators; at head only `fuel-onnx` is allowlisted. `02` diagram marks existing crates absent.
9. `05` compliance table and tier listing (section 5). `10-decisions-log`: `FusionMissRecord`/`NoBackendKernel` names do not exist (`MissRecord` does); "~36 Op variants" is far below the current enum; "bundled baseline Judge" absent.
10. `fuel-graph/src/lib.rs:1162` `Op::Branch` comment ("nothing constructs it yet") is stale.
11. `03-ir` 74 "single carry" (code has `n_carries`); 76/79 early-exit driver and Hopfield consumer (absent).
12. ROADMAP's whole-graph `Arc<Graph>` transaction wording: corrected in #687.

## 7. Not verified (stated by the passes, kept)

Live branch count on a real model (needs a GPU run); `decompose` totality over every fused op; backend-internal
fusion/caching (05 "rules out"); kernel-revision-hash consumer; `Op::Release`/alias ordering for bundles;
whether ONNX `LayerNormalization` emits `Op::Fused`; shape-oracle, COW binding table, two-axis decline entries of
the decisions log; Vulkan overlap correctness; every test name (not run).

## 8. Plan, ordered by how base-layer the gap is

0. **Row 157 / GAP-385: multi-node placement pathfinder** (design note, then red tests on a fan-out and a
   residual join). This produces the multi-path DAG the architecture is built around. First.
1. **Alternatives beyond placement** as decision points: kernel-variant carried in the graph (stamp the kernel,
   not just the backend), fusion-vs-decomposition from the registry, algebraic. Requires wiring `RuleRegistry`
   and the retained base map into realize (the "two-stage transformation" of `04`).
2. **Retained base map + regional transactions and swap + load-time planning** (rows 88/89; a decision-point
   region is the unit). Needed before hooks and for the TTFT promise.
3. **Executor scheduler**: runtime ready-set and slots, bounded lookahead, `RouteCache` and `LOOKAHEAD_K` wired
   or deleted, memory-pressure serialization, cancellation (reconcile 06 vs 15).
4. **Whole-path frontier, wall-clock cost, coupling** (consumers of 1).
5. **Persistence** (`11`): graph serializer, plan cache, fingerprint, snapshot; then bundled/online Judge.
6. **Tolerance budgets (07) and algebraic rewrites on realize**; pattern harvest (08) after a decision on its
   privacy posture (CIRESNAVE CALL, rows 86/87).
7. **Backend contract compliance**: implement or delete the unbuilt traits, fix the compliance table, remove the
   `fuel-dispatch` dependency from the aocl/mkl crates (ARC), Metal fast_math default.
8. **Doc corrections** in section 6, landing with each row so the docs stop promising what is absent.

Hooks (tensor tap/inject) sit on top of steps 0 to 2 and are not scheduled by this note.
