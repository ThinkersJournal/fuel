// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors file reading — the mmap/view surface.
//!
//! Moved here from `fuel-core` (fuel-core dissolution Slice 3,
//! `docs/session-prompts/fuel-core-dissolution-b1.md`). The `fuel-core` re-export shim was
//! retired (fuel-core dissolution, GAP-347 PR 5): direct consumers were repointed here, and
//! `fuel::safetensors` is re-exported straight from this module by the `fuel` facade.
//!
//! - [`MmapedSafetensors`] — memory-maps one or more files and hands out
//!   `safetensors::TensorView`s via `get`/`tensors`. This is what the lazy stack uses: it reads
//!   `view.data()` + `view.dtype()` and decodes the bytes itself (`Tensor::from_safetensors_view`
//!   in `fuel-core/src/lazy.rs`, and the per-model `from_hub`/`from_gguf` constructors in
//!   `fuel-model-llama`/`fuel-model-phi`/`fuel-transformers`).
//! - [`BufferedSafetensors`] — the same, over an owned `Vec<u8>`.
//!
//! The EAGER half was removed in B6: `impl st::View for Tensor`, the `Load`
//! trait and its `.load(device)` method, every `convert*` helper, the free
//! `load`/`load_buffer`/`save` functions, and `SliceSafetensors`. All of them
//! produced or consumed the eager `Tensor`, and nothing lazy called them —
//! verified: zero `.load(` call sites workspace-wide.

pub use fuel_formats::safetensors::{BufferedSafetensors, MmapedFile, MmapedSafetensors};
