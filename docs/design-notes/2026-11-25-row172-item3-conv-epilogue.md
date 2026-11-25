# Row 172 item 3: a fused epilogue on the 2-D convolution (design)

Status: DESIGN, 2026-11-25, lane Fuel 1, branch `row172-item2b`. Nothing here is built.
Source: Unpopped `docs/ir-union-asks.md` 1b; survey row CUTLASS 51 (the conv fused epilogue: bias,
activation, per-channel scale). Companion to the item 1 and item 2 notes (same registration recipe).

## What exists today (read at this branch)

- `FusedOps::CONV2D` (id 6, `fuel-graph/src/registry/conv2d.rs`): `Conv2D { stride, padding, groups }`,
  inputs `(x, weight)` or `(x, weight, bias)`; `decompose` lowers any `groups >= 1` to the im2col
  recipe with a broadcast-`Add` bias tail; CPU kernels with and without bias are registered
  (`cpu/conv.fkc.md`, the `[T,T,T]` and `[T,T,T,T]` keys). NO activation and NO scale in the op.
- The primitives for the epilogue exist: `Relu`, `Gelu`, `Silu`, `Mul` (and `BroadcastTo`).
- Baracuda (read-only, `origin/main` `OP-MATRIX.md`): the CUTLASS plans' epilogue family is the bias
  family `{Identity, Bias, BiasRelu, BiasGelu, BiasSilu}`; there is no per-channel-scale epilogue
  (see the finding sent 2026-11-25: row-scale and TopK epilogues are not in a GEMM plan).

## Decisions

1. **One new fused op, not a flag on `Conv2D`.** `FusedOps::CONV2D_EPILOGUE` (id 29),
   `FusedOpParams::Conv2dEpilogue { stride, padding, groups, act, has_scale }`, where `act` is a closed
   enum `{ None, Relu, Gelu, Silu }`. Reason: `FusedOpParams::Conv2D` is a CSE/identity key and a
   BREAKING field add would touch every constructor; a sibling op leaves `Conv2D` byte-identical and
   follows the item 1/2 pattern. Inputs: `(x, weight)` ++ `bias`? ++ `scale`? in that order, bias
   present iff the op was built with one (the input count disambiguates, as `Conv2D`), scale
   `[Cout]`-shaped (per output channel) iff `has_scale`.
2. **Math, fixed order** (the recipe): `y = act((conv(x, w) + bias) * scale)`. Bias before the scale
   is the BatchNorm-fold order (`scale` carries `gamma / sqrt(var + eps)`); the order is part of the
   op's identity and is pinned by a test that swaps it.
3. **Recipe/decompose**: `decompose_via_recipe` over `Conv2D`'s existing recipe plus the epilogue
   tail (`Add` bias, `Mul` scale broadcast over `[N, Cout, H, W]` from `[Cout]` via a reshape to
   `[1, Cout, 1, 1]`, then the activation primitive): total, never panic (G2). Canonical pattern:
   `act(Mul(Add(Conv2D(..), bias), scale))` with each intermediate single-consumer, and the
   degenerate forms (`bias` only, `act` only) matching the SAME op with the absent parts omitted, so a
   plain `Relu(Conv2D)` also fuses.
4. **Where roles are declared**: as items 1 and 2, the conv's contraction roles are not yet declared
   by the op (item 6b waits on KISS's keys for more than one K axis; a conv's K axis is the
   `Cin/g * Kh * Kw` im2col contraction). This item adds NO key facts; it only carries the epilogue.

## What binds

- **CPU**: new byte kernels for the four float dtypes: the existing conv kernel, then a fused tail
  loop (bias, scale, activation) in the same dtype/accumulator discipline the item 1 kernel uses
  (conv result narrowed to the element dtype first, then scaled, then activated: the recipe's
  rounding order). Contracts, probes, ledger append, predicted pin deltas as items 1 and 2.
- **CUDA**: baracuda's `BiasRelu/BiasGelu/BiasSilu` cover the bias+activation half of a conv-as-GEMM
  only if the conv is lowered to a GEMM first; the per-channel scale has no epilogue. Binding is
  therefore the composed form until baracuda adds epilogues (decision for pm-kernels, see item 2).

## Order of work (each its own commit, red first)

1. Red tests: pattern (matches the 4 forms, declines a shared conv output and a scale of the wrong
   length), shape/dtype rules, order-swap sabotage. 2. Entry/params/id/recipe/decompose. 3. CPU
   kernels, contracts, probes, ledger, pins, e2e realize test vs the hand conv + epilogue. 4. CUDA
   after the baracuda decision.

## Open questions

- Which activations does Unpopped's IR need beyond `{Relu, Gelu, Silu}` (e.g. `Tanh`, `Sigmoid`,
  `LeakyRelu` with a slope)? The closed enum grows by adding a variant (a BREAKING add on a
  pre-1.0 enum, bundled with a release).
- Is the per-channel scale ever a per-(channel, group) or per-pixel tensor? Assumed `[Cout]`.
- Estimate: 2 to 3 lane-days, as the row says (the CPU half; CUDA depends on the baracuda decision).
