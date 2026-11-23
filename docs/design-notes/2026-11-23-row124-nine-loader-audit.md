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
