---
fkc_version: 1
provider:
  name: fuel-cuda-backend
  backend: Cuda
  kernel_source: "baracuda"
  link_registry: fuel_cuda_backend::fkc::ENTRY_POINTS
  revision_base: "git:f41137b4"
---

# fuel-cuda-backend - TopKLastDim (baracuda block-bitonic topk) kernel contract

CUDA (baracuda `baracuda_kernels_topk_{f32,f64}_run`). Two sections, one per dtype, each its own key
`[T, T]` (the input then the bundle's primary slot). See each section for the caps and what is NOT claimed.

---

## top_k_last_dim_f32  (top-k values + U32 indices along the last axis, f32; baracuda block-bitonic)

TopKLastDim on CUDA (ROADMAP row 172 item 2) via baracuda's standalone `topk` (`baracuda_kernels_topk_f32_run`, `TopkPlan<f32>`). One input `x [.., n]`; ONE bundled output `[values [.., k] (f32) ; pad to 4 bytes ; indices [.., k] U32]`. **Caps (baracuda trailblazer): `n <= 1024`, `k <= 64`, f32/f64 only.** A shape outside the caps is a typed `Err` before any allocation or launch: the binding table admits a kernel by dtype/layout/rank, not by value, so there is NO decline to the CPU kernel for an over-cap shape (f16/bf16 inputs have no CUDA binding and resolve to the CPU kernel by dtype). **NaN and ties follow the `torch.topk` convention (the CPU kernel's rule): a NaN is the LARGEST value, ties take the LOWER index first.** This holds from baracuda 0.1.0-alpha.188 (the top-k NaN/ties fix, baracuda #429); at alpha.186 it did not (ties picked `[1,2,7]` for `[3,5,5,1,5,2,4,5]` k=3, and a NaN row lost its 9). Measured live (RTX 4070, 2026-11-25): the two pinned vectors and 3840 random NaN/tie rows (10 shapes up to n=1024, k=64) equal the CPU kernel, pinned by `baracuda_topk_live.rs` so a baracuda regression reddens it (GAP-402). The over-cap error and the absent per-shape decline are GAP-401 (provisional).

```fkc
kernel: top_k_last_dim_f32
op_kind: TopKLastDim
blurb: "Top-k values + U32 indices along the last axis (f32; baracuda block-bitonic; n<=1024, k<=64); bundled output [values; indices]."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::top_k_last_dim_f32"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F32]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
  op_params:
    variant: TopKLastDim
    fields:
      rows: { kind: usize, constraint: "== product of x.dims[..-1]" }
      n:    { kind: usize, constraint: "== x.dim[-1] <= 1024" }
      k:    { kind: usize, constraint: "1 <= k <= min(n, 64)" }

return:
  bundle:
    - { name: values,  dtype_rule: passthrough(x), shape_rule: from_params(values),  layout_guarantee: contiguous }
    - { name: indices, dtype_rule: fixed(U32),     shape_rule: from_params(indices), layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "rows * n * log2(n)"
  bytes_moved: "(rows * n + 2 * rows * k) * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "baracuda documents the fixed comparator network as deterministic on identical hardware; Fuel has not earned a verification-ledger record for it (no live run recorded), so the claim is NOT made here."

determinism: same_hardware_bitwise
```

---

## top_k_last_dim_f64  (top-k values + U32 indices along the last axis, f64; baracuda block-bitonic)

TopKLastDim on CUDA (ROADMAP row 172 item 2) via baracuda's standalone `topk` (`baracuda_kernels_topk_f64_run`, `TopkPlan<f64>`). One input `x [.., n]`; ONE bundled output `[values [.., k] (f64) ; pad to 4 bytes ; indices [.., k] U32]`. **Caps (baracuda trailblazer): `n <= 1024`, `k <= 64`, f32/f64 only.** A shape outside the caps is a typed `Err` before any allocation or launch: the binding table admits a kernel by dtype/layout/rank, not by value, so there is NO decline to the CPU kernel for an over-cap shape (f16/bf16 inputs have no CUDA binding and resolve to the CPU kernel by dtype). **NaN and ties follow the `torch.topk` convention (the CPU kernel's rule): a NaN is the LARGEST value, ties take the LOWER index first.** This holds from baracuda 0.1.0-alpha.188 (the top-k NaN/ties fix, baracuda #429); at alpha.186 it did not (ties picked `[1,2,7]` for `[3,5,5,1,5,2,4,5]` k=3, and a NaN row lost its 9). Measured live (RTX 4070, 2026-11-25): the two pinned vectors and 3840 random NaN/tie rows (10 shapes up to n=1024, k=64) equal the CPU kernel, pinned by `baracuda_topk_live.rs` so a baracuda regression reddens it (GAP-402). The over-cap error and the absent per-shape decline are GAP-401 (provisional).

```fkc
kernel: top_k_last_dim_f64
op_kind: TopKLastDim
blurb: "Top-k values + U32 indices along the last axis (f64; baracuda block-bitonic; n<=1024, k<=64); bundled output [values; indices]."
backend: Cuda
kernel_source: "baracuda"
entry_point: "fuel_cuda_backend::fkc::top_k_last_dim_f64"
kernel_revision_hash: auto

accept:
  inputs:
    - name: x
      dtypes: [F64]
      layout: { contiguous: required, strided: rejected, broadcast_stride0: rejected, start_offset: rejected, reverse_strides: rejected }
      rank: "1.."
  op_params:
    variant: TopKLastDim
    fields:
      rows: { kind: usize, constraint: "== product of x.dims[..-1]" }
      n:    { kind: usize, constraint: "== x.dim[-1] <= 1024" }
      k:    { kind: usize, constraint: "1 <= k <= min(n, 64)" }

return:
  bundle:
    - { name: values,  dtype_rule: passthrough(x), shape_rule: from_params(values),  layout_guarantee: contiguous }
    - { name: indices, dtype_rule: fixed(U32),     shape_rule: from_params(indices), layout_guarantee: contiguous }

caps:
  awkward_layout_strategy: requires_contiguous
  in_place: false
  alignment_bytes: 64
  access_granularity_bits: 32

cost:
  provenance: judge_measured
  class: reduction
  flops: "rows * n * log2(n)"
  bytes_moved: "(rows * n + 2 * rows * k) * dtype_bytes"

precision:
  bit_stable_on_same_hardware: false
  max_ulp: ~
  max_relative: ~
  max_absolute: ~
  audited: false
  notes: "baracuda documents the fixed comparator network as deterministic on identical hardware; Fuel has not earned a verification-ledger record for it (no live run recorded), so the claim is NOT made here."

determinism: same_hardware_bitwise
```
