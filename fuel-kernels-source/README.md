# fuel-kernels-source

Kernel source files (.glsl / .wgsl) shared across Fuel GPU backends. Compile ahead of time with kernels/compile.sh; compiled artifacts live in fuel-vulkan-kernels (SPIR-V) and fuel-cuda-kernels (PTX / cubin).

Part of the [Fuel](https://github.com/ThinkersJournal/fuel) workspace: a DAG-first, lazy-only ML framework for Rust. Licensed under MIT OR Apache-2.0.

## Rebuilding and checking a GLSL kernel

One command per kernel, from the repo root: `VULKAN_SDK=<sdk> bash fuel-kernels-source/kernels/compile.sh <name>.glsl`
(writes `fuel-vulkan-kernels/spv/<name>.spv`). Pinned toolchain: `glslangValidator` 16.2.0 (Vulkan SDK 1.4.350.0),
`-V -S comp --target-env vulkan1.1`. The output is byte-reproducible with that version: rebuilding the committed
`qmatvec_q5_k.glsl` and `matvec.glsl` leaves `git status` clean. Check: rebuild, then `git diff --exit-code fuel-vulkan-kernels/spv`.

`qmatvec_q5_k.glsl`, `qmatvec_q3_k.glsl`, `qmatvec_iq4_nl.glsl`, `qmatvec_iq4_xs.glsl` and `qmatvec_iq3_s.glsl` (ROADMAP row 119) are checked on a device by `vulkan_dispatch_qmatmul_q4_km` (its Q5_K block, exact f64 oracle
against `BlockQ5K::to_float`, N = 16 and 15, m = 1 and 3). The seeded misread to rerun after a shader edit: replace
`(((h & u1) != 0u) ? 16u : 0u)` with `0u` in the Q5_K shader (drops the high-bit plane), `? 4 : 0` with `? 0 : 0` in the Q3_K shader (drops the inverted high-bit term) or `KV[q >> 4u]` with `KV[q & 15u]` in the IQ4_NL shader (reads the high nibble as the low one), or `uint sh = 2u * ib;` with `uint sh = ib;` in the IQ4_XS shader (misreads the high scale bits), or `(8u - 2u * l)) & 256u` with `(8u - 2u * l)) & 0u` in the IQ3_S template and regenerate (drops the 9th grid-index bit), rebuild, and that test must fail.
