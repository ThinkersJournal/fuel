# Row 124 step 2: do the nine other GGUF loaders get QMatMul on Vulkan and weight residency?

Date: 2026-11-23. Lane: fuel. Method: READ ONLY, at `origin/main` `7ecc79f6e` (`git show` of the files named
below). Nothing was built or run for this note; every statement is about source at that ref, and none is a
measurement. The residency half refers to the branch `weight-residency` at `9d9b68cfa`, which is not on main.

## What the question names

`566bcf7f1` (#534, row 124 step 2) made GGUF weights stay quantized in nine more loaders, after the Qwen3
loader. The nine, by the files that call `WeightStorage::from_ggml_bytes` at main: `lazy_quantized_gemma3`,
`_glm4`, `_lfm2`, `_llama`, `_phi3`, `_qwen2`, `_qwen3_moe`, `_smollm3`, `_t5`. (`lazy_quantized_qwen3` is the
tenth call site and the one measured; `lazy_qwen3_5_gguf` also calls it through its own loader and is not one
of the nine.)

## Finding 1: the ten call sites are one construct

Each of the ten has the identical block: the `_ =>` arm of the dtype match, the same comment, and
`WeightStorage::from_ggml_bytes(dt, bytes, in_features, out_features)`. Whatever the Qwen3 example shows for
`WeightStorage::Ggml` therefore applies to the other nine at the loader level; the models differ only in what
else they build around it. Checked by `git grep -n -B6 from_ggml_bytes` over the ten files (same output shape in
all ten). NOT checked: the code AFTER that call in each loader, nor any model's graph.

## Finding 2: residency (row 172) reaches them without a per-loader change

`WeightStorage::Ggml` builds its graph `Const` through `const_u32_like(Arc::clone(words), ...)`. On
`weight-residency` that constructor records a `ConstOrigin` (`ConstOrigin::of(&v, DType::U32)`), and the
residency cache keys on that origin. So the nine loaders are covered by the same mechanism as Qwen3, provided
the model keeps its `WeightStorage` alive (the pin succeeds only while the source `Arc` lives). NOT measured on
any of the nine: row 172's measurement is Qwen3-0.6B Q4_K_M (plus the TinyLlama Q4_0 swap) only.

## Finding 3: `from_ggml_bytes` admits more types than Vulkan can run

`from_ggml_bytes` admits Q8_0, Q4_K, Q6_K, Q3_K, Q5_K, IQ4_NL, IQ4_XS and IQ3_S. `qmatmul_vk` in
`fuel-vulkan-dispatch/src/vulkan_dispatch.rs` has arms for `Q4_0`, `Q4_K_M`, `Q8_0` and `Q6K` only; every other
`QuantType` returns `Err("... not wired on Vulkan ...")`. The Vulkan binding is keyed on dtypes
`[F32, U32, F32]` and not on the quant type (`register_vulkan_kernels_binds_qmatmul_from_contract`), so the
kernel table alone does not distinguish a Q5_K weight from a Q4_K one.

Consequence to settle, NOT established here: for a Q3_K, Q5_K, IQ4_NL, IQ4_XS or IQ3_S GGUF, either the placer
sends the `QMatMul` to the CPU arm (slow, memory-resident, row 168's territory) or it places it on Vulkan and
the call fails at run time. Which one happens was not run. The row 119 text says those types "go to the CPU",
which predates the contract binding. A K-quant mix file (Q4_K_M typically carries Q6_K and Q4_K, but Q3_K_M and
Q5_K_M files carry Q3_K and Q5_K) is the input that would show it.

## What would settle it

1. Per loader, a synthetic one-layer model with each admitted type, placed on Vulkan, count `QMatMul` launches
   by arm (the `qwen3_vulkan_profile` counters), for Q4_K, Q6_K, Q8_0 (wired) and Q3_K, Q5_K (not wired).
2. Residency on one non-Qwen loader (llama, the TinyLlama Q4_0 swap already covers the Q4_0 variant, not
   `Ggml`): second same-context prefill with `FUEL_WEIGHT_RESIDENCY=0|1`, logits hash equal.
3. If (1) shows a run-time error for an unwired type, the loaders should decline to keep that type quantized
   on a Vulkan device or the kernel should be written (row 119); a registered decision in `docs/gaps.md`
   either way.

## RESULT of the probe (added 2026-11-23; measured)

Build: `fuel-transformers` example `gguf_unwired_quant_probe` (this branch, `--release`, `--features
dev-tools,vulkan`, `-j 4`), tree = `7ecc79f6e` plus this branch's docs, example, `files_with` helper. Run once
(n = 1 process) through `gpu-run`, RTX 4070 Laptop (Vulkan adapter 1), backlog feed at its default (OFF).
Per type, four realizes: a bare one-op `qmatmul` graph at `m=1 n=256`, `m=1 n=255` (odd), `m=3 n=256`, all
`k=256`, and the one-layer llama of `tests/common/gguf_synth.rs` (3 tokens), weights resident as blocks
(`resident_bytes` equals the quantized file bytes in every model run). Outcome (a) = ok with every `QMatMul` on
Vulkan, (b) = ok with a `QMatMul` on the CPU arm, (c) = error.

| type | role | isolated 256/m1, 255/m1, 256/m3, model | `QMatMul` launches on Vulkan / CPU |
|---|---|---|---|
| Q4_K | control | A A A A | 1 / 0 each; model 8 / 0 |
| Q6_K | control | A A A A | 1 / 0 each; model 8 / 0 |
| Q3_K | subject | C C C C | the op was placed on Vulkan, 0 on CPU |
| Q5_K | subject | C C C C | same |
| IQ4_NL | subject | C C C C | same |
| IQ4_XS | subject | C C C C | same |
| IQ3_S (synthetic bytes) | subject | C C C C | same |

Every (c) is the same error, e.g. `vulkan_dispatch::qmatmul: QuantType Q5K not wired on Vulkan; add a
dequant-then-matmul path or a fused kernel and retry`. The planner puts the `QMatMul` on Vulkan and the kernel
refuses: it is NOT a CPU fallback (no `QMatMul` launch on the CPU arm in any run), it is a run-time error.
Controls came out (a) everywhere, so the probe can see a success. A model whose GGUF holds any of these five
types fails to realize on Vulkan.

**First bad commit, by source, NOT by running the probe on its parent:** `git log -S` on
`fuel-tensor/src/lazy.rs` puts the arms for all five types in `ed3ce9746` (#686, 2026-11-15, 0.49.0, "Q3_K/Q5_K/IQ
stay quantized"), not in #534 (`566bcf7f1`), which admitted Q8_0/Q4_K/Q6_K. Before #686 these types
dequantized to f32 in the loader and ran as an f32 MatMul on Vulkan. The pre-#686 behaviour was not re-run.

**Not tested:** CUDA (it may wire these types differently); a real Q5_K_M file (the synthetic model is one layer
with `H = 256`); the CPU device (#686 added a CPU float route for these types); m larger than 3.
The exe exited 139 (segmentation fault) AFTER printing its last line; the cause is not investigated (the
verdict lines were all printed before it).

Options for the PM (no fix made): (1) `from_ggml_bytes` declines these five types on a device with no kernel for
them, so the loader dequantizes to f32 as before #686; (2) wire Vulkan kernels (row 119 for Q3_K/Q5_K, new
work for IQ4_NL/IQ4_XS/IQ3_S); (3) place the `QMatMul` on the CPU arm for an unwired type (the CPU has the #686
float route) so it runs, slowly.

## What was done (hotfix, PM ruling 2026-11-23)

The example behind the RESULT table lives on branch `row124-audit` at `18a2c897c`; this change replaces it with the
test `fuel-transformers/tests/gguf_unwired_quant_vulkan_live.rs` and applies option (1) in the process-wide form:
`from_ggml_bytes` has no device (neither do the ten loaders), so in a build with the `vulkan` feature it
declines every type `qmatmul_vk` has no arm for, whatever device later runs the model. The per-device decision
(the loaders taking the target) and the kernels are the follow-up (GAP-397). Option (3) in the list above was
rejected for cost: running an unwired type through a host dequantize reads the whole weight back per matmul.
Measured with the test (n = 1 process): red without the decline (Q3_K fails to realize), green with it (7 of 7
types realize; Q4_K and Q6_K unchanged at 8 of 8 `QMatMul` on Vulkan); the f32 residency of the five types is
5.8x (Q5_K) to 9.3x (Q3_K, IQ3_S) the block bytes on the synthetic projections.
