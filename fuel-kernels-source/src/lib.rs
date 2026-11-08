// SPDX-License-Identifier: MIT OR Apache-2.0
//! # fuel-kernels-source
//!
//! Home for Fuel's remaining GPU kernel source files (`.glsl`, `.wgsl`)
//! and the `compile.sh` script that turns them into backend artifacts.
//!
//! This crate intentionally has no Rust code. Downstream consumers
//! read the compiled outputs from:
//!
//! - [`fuel-vulkan-kernels`](../fuel_vulkan_kernels/) — SPIR-V baked
//!   into the binary via `include_bytes!`.
//! - `fuel-cuda-kernels` — PTX/cubin (today `.cu` files are
//!   co-located there; Slang→CUDA unification is planned but not yet
//!   wired).
//!
//! ## Layout
//!
//! - The Slang sources are no longer here: they moved to the published
//!   `fishstringer-kernels-source` crate (0.46.0; ROADMAP row 143 covers
//!   the `.glsl` files next).
//! - `kernels/*.wgsl` — legacy WGSL sources (being hand-ported to
//!   Slang; see todo list).
//! - `kernels/*.glsl` — legacy GLSL sources (kept for the few kernels
//!   where GLSL is more ergonomic, e.g., cooperative-matrix matmul).
//! - `kernels/compile.sh` — ahead-of-time compiler invocation, writes
//!   SPIR-V to `../fuel-vulkan-kernels/spv/`.
//!
//! ## Why a separate crate
//!
//! Keeping the sources in their own crate makes the source→artifact
//! split explicit: the source file is the authoritative
//! implementation; the `.spv` in fuel-vulkan-kernels and the future
//! CUDA in fuel-cuda-kernels are derived. Tooling that wants to
//! regenerate artifacts (CI, future cross-backend compiler) has a
//! single directory to iterate over.
