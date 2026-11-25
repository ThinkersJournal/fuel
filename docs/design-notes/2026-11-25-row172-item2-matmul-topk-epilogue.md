# Row 172 item 2: a TopK epilogue on the matmul (design)

Status: DESIGN, 2026-11-25, lane Fuel 1, branch `row-union-contraction`. Nothing here is built.
Source: Unpopped `docs/ir-union-asks.md` 1b; survey row CUTLASS 38 (TopK in the epilogue; softmax
stays inside the attention kernel). Companion to
`2026-11-25-row172-item1-row-scaled-matmul.md` (the recipe/registration pattern reused here).

## What exists today (read at this branch)

- **No TopK op in the graph.** `git grep -i topk` over `fuel-ir`, `fuel-graph`, `fuel-dispatch`
  `src/` finds only a comment in `fuel-dispatch/src/kernel.rs` ("multi-output ops (topk, ...)") and
  the HOST-side sampler `fuel-tensor/src/sampling.rs` (`top_k` over an already-realized logits
  vector). The MoE plan names `Op::TopKRoute` (`docs/frontier-architecture-gaps.md`) as planned, not
  built. So the epilogue has no op to fuse INTO: item 2 is two steps, not one.
- **Multi-output fused ops exist**: `SELECTIVE_SCAN` and `SSD_CHUNK_SCAN` carry
  `output_views: Some(..)` (slot 0 primary, `shape_rule`/`dtype_rule` report slot 0).
- **Index precedent**: `ArgMaxDim` yields `U32` indices, reduced dim removed.

## Decisions

1. **Step A, the standalone op.** `FusedOps::TOP_K_LAST_DIM` (next free id after 26 = 27),
   `FusedOpParams::TopKLastDim { k: usize }`, multi-output: slot 0 `values [.., k]` (input dtype),
   slot 1 `indices [.., k]` (`U32`, like `ArgMaxDim`). Descending by value; **ties take the lower
   index first** (the same rule `ArgMaxDim` pins); NaN follows the pinned NaN conventions
   (`nan-conventions-pinned`: CPU, CUDA and Vulkan agree), to be re-read before the kernel is
   written, not assumed here. `k >= 1` and `k <= N` validated at graph-build time (the Result,
   never a panic).
2. **Step A decompose** is IMPERATIVE (like the conv recipes), not portable data, because `k`
   unrolls: `k` rounds of `ArgMaxDim` -> `Gather` the value -> mask the winner (`MaskedFill` with
   the dtype's lowest) -> concat along the last axis. Total, never-panic (G2: returns the node on any
   failure). A static `k` is required; a data-dependent `k` is out of scope.
3. **Step B, the epilogue.** `FusedOps::MATMUL_TOP_K` (id 28), `FusedOpParams::MatMulTopK { k }`,
   inputs `(a, b)`, outputs as step A over the matmul's `N`. Canonical pattern: `TopKLastDim(MatMul(a,
   b))` where the MatMul has no other consumer (same single-consumer rule as items 1 and FusedLinear).
   Decompose = `TopKLastDim(MatMul)`. So the fused node is a rewrite of step A's node, and the CPU
   path needs NO new kernel for B beyond composing the two (a CPU `MatMulTopK` kernel is a later
   performance item); the correctness reference is step A's decompose.
4. **Where roles are declared.** As item 1: the MatMul inside declares M/N/K/batch
   (`telemetry/contraction_facts.rs`); the epilogue adds no contraction role, only an output
   reduction over N, so `StructureKey` carries `k` as a declared epilogue fact (KISS owns the key
   spelling; this lane only carries it). Not key-affecting until KISS says so.

## What binds

- **CPU**: step A gets byte kernels (4 float dtypes; selection by a stable partial sort, k-ascending
  order fixed so the result is bit-stable). Step B registers with the composed form first.
- **CUDA**: Baracuda's CUTLASS 38 epilogue is the performance path. Entry point and ABI NOT read yet
  (sibling repo, read-only); read together with CUTLASS 35 (item 1) in one pass.
- Vulkan/Metal: the decompose serves them.

## Order of work (each its own commit; red first)

1. Red tests: step A pattern/shape rules (`k > N` declined at build time; values+indices shapes;
   tie order; NaN per the pinned convention) against a host reference. 2. Step A entry/params/id,
   decompose, round-trip. 3. Step A CPU kernel + FKC contracts + probe + ledger append (the item-1
   recipe: `cpu/` contract, `fused/` contract, `OpKind`, wrappers, pins moved by PREDICTED deltas).
4. Step B pattern, entry, registration. 5. CUDA binding after the Baracuda read.

## Resolved questions (2026-11-25, before the red tests)

- **NaN order.** The 2026-07-08 NaN pin (`docs/architecture/10-decisions-log.md`, entry "Relu, Maximum and
  Minimum are NaN-propagating", and the memory `nan-conventions-pinned`) is ELEMENTWISE ONLY: it states
  that reductions and softmax internals are "explicitly untouched". So no Fuel pin covers TopK. Decision
  by the standing norm "match external convention (PyTorch/CUDA)": follow `torch.topk`, **NaN sorts as
  the largest value** (NaNs come first in the descending output, lowest index first among NaNs). This is
  a NEW decision for this op, recorded in the decisions log with the op's commit, not an inherited pin.
  Ties between equal non-NaN values: lower index first (deterministic; torch leaves it unspecified).
- **`k` is static.** A node's output shape is `[.., k]`, and Fuel plans once over static shapes; a
  data-dependent `k` is a data-dependent output shape (the shapes keystone, `Op::NonZeroIndices`
  territory) and is out of scope. `k` lives in `FusedOpParams::TopKLastDim { k }`, so it is part of the
  node's CSE/recipe identity.
- **Index dtype `U32`**, as `ArgMaxDim` yields. An `I64` consumer casts.
- Still to confirm with Unpopped (does not block): whether their IR ever needs a dynamic `k`.
