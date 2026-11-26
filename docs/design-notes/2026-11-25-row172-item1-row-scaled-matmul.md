# Row 172 item 1: a per-row `[M]` scale epilogue on the matmul (recipe design)

Status: DESIGN, 2026-11-25, lane Fuel 1, branch `row-union-contraction`. Nothing here is built yet.
Source: Unpopped `docs/ir-union-asks.md` 1b; survey rows CUTLASS 35 and ThunderKittens 35.

## Why a new recipe, not a planner flag

`fuel-graph/src/registry/fused_linear.rs` `canonical_pattern` matches only
`Add(MatMul(a, b), bias_broadcast)` with a rank-1 `[N]` bias. A per-token (per-row) scale
`Mul(MatMul(a, b), BroadcastTo(scale))`, `scale` of shape `[..batch, M, 1]`, matches no registry
pattern, so it is never fused and no kernel can be offered for it. There is no decline to relax.

## Decisions

1. **Name and id.** `FusedOps::ROW_SCALED_MATMUL`, the next free `FusedOpId` after `AWQ_MATMUL` (25),
   registry name `"RowScaledMatMul"`, family `Forward`.
2. **Inputs.** `a [..batch, M, K]`, `b [..batch, K, N]`, `row_scale [..batch, M, 1]`. The scale
   arrives in keepdim form, so the recipe needs no reshape (a rank-polymorphic `[..,M] -> [..,M,1]`
   is not expressible in `ShapeExpr`: `Dims` is fixed-arity). Output `[..batch, M, N]`, dtype of `a`;
   all three dtypes must agree (same rule as FusedLinear).
3. **Recipe (portable data, parameterless).**
   `Mul(MatMul(a, b), BroadcastTo(WithDim{op0, LAST, Extent{op1, LAST}})(scale))`. The broadcast
   target is `a`'s shape with its LAST axis (K) replaced by `b`'s LAST extent (N), the matmul output,
   exactly the FusedLinear target; only the broadcast axis differs (`[..,M,1] -> [..,M,N]`).
4. **Total, never-panic `decompose`** through `decompose_via_recipe` (G1/G2/G3): a wrong payload or
   a resolution decline returns the node unchanged. Backward: `NotDifferentiable` first (as
   FusedLinear), the 3-grad arm is a later step if a training consumer appears.
5. **Canonical pattern** recognizes `Mul(MatMul(a, b), bc)` in BOTH operand orders (Mul is
   commutative) where `bc` is a `BroadcastTo` of a tensor whose last extent is 1 and whose other
   extents equal `a`'s leading shape through M; bindings `[(0,a),(1,b),(2,scale)]`. Single-consumer
   rule as FusedLinear (the MatMul must be consumed only by the Mul, else fusing duplicates it).
   A `[N]`-last bias (non-1 last extent) must NOT match: a sabotage sibling test pins that.
6. **Where it lives.** `fuel-graph` only for the recipe, entry, id and pattern; nothing in
   `fuel-ir`. The MatMul inside already declares its roles (slice 1,
   `fuel-dispatch/src/telemetry/contraction_facts.rs`); the fused node gets its own
   `matmul_declared_facts` read (same M/N/K, batch) when the planner keys it.

## What binds (kernels)

- **CPU reference:** a fallback is free: with no registered kernel the recipe lowers to
  `MatMul` + `Mul`, which is the byte-reference.
- **CUDA:** the kernel layer already provides the epilogue (CUTLASS 35). Binding it is an FKC
  contract in `fuel-cuda-dispatch/kernel-contracts/cuda/` plus a link entry; which Baracuda entry
  point and its argument ABI is the open fact (NOT yet read: sibling repo, read-only).
- **Vulkan/Metal:** not in scope for this slice; the recipe lowering serves them.

## Order of work (each its own commit)

1. This note. 2. Red test: `canonical_pattern` matches the scaled form, declines the `[N]` bias form
and the shared-MatMul form. 3. Entry, id, recipe, pattern, shape/dtype rules, registry
registration; the `fuse` round-trip test (lower then fuse recovers the node). 4. Numeric oracle:
fused-lowered vs hand `matmul * scale`, including a batched case. 5. CUDA contract (needs the
Baracuda read, then a live run under `gpu-run`).

## Open questions (none block steps 1-4)

- Does Unpopped's IR carry the scale as `[M]` or `[M,1]`? Slice assumes `[..,M,1]`; an `[M]`
  adapter belongs in the Unpopped-facing lowering, not the recipe. To confirm with pm-kernels.
- The Baracuda entry point name and ABI for step 5.
