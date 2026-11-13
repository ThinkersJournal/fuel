// SPDX-License-Identifier: MIT OR Apache-2.0
//! Object-safe backend traits for dynamic dispatch.
//!
//! [`DynBackendStorage`] and [`DynBackendDevice`] are the object-safe
//! contract surface. `DynBackendStorage` mirrors
//! [`BackendStorage`](crate::backend::BackendStorage); the generic
//! `BackendDevice` trait `DynBackendDevice` once mirrored was DELETED in
//! the 2026-04-30 backend-agnostic refactor (see the note on
//! [`crate::backend`]), and backends implement it directly. They replace `Self`
//! return types with `Box<dyn DynBackendStorage>`, eliminate generic type
//! parameters, and drop the `Sized` bound so that `dyn DynBackendStorage`
//! and `dyn DynBackendDevice` are legal trait objects.
//!
//! These traits are the primary interface for all backends in fuel.
//! Every backend — built-in or third-party — implements these traits, and
//! `Device` and `Storage` are newtype wrappers around `Arc<dyn DynBackendDevice>`
//! and `Box<dyn DynBackendStorage>` respectively.
//!
//! # Implementing a custom backend
//!
//! ```rust,ignore
//! use fuel_backend_contract::dyn_backend::{DynBackendDevice, DynBackendStorage};
//! use std::sync::Arc;
//!
//! struct MyDevice { /* ... */ }
//! impl DynBackendDevice for MyDevice { /* ... */ }
//!
//! struct MyStorage { /* ... */ }
//! impl DynBackendStorage for MyStorage { /* ... */ }
//! ```
use fuel_ir::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use fuel_ir::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use fuel_ir::{DType, DeviceLocation, Error, GgmlDType, HostBuffer, Layout, Result, Scalar, Shape};
use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

/// Object-safe counterpart of [`BackendStorage`](crate::backend::BackendStorage).
///
/// Every method mirrors the corresponding `BackendStorage` method but returns
/// `Box<dyn DynBackendStorage>` instead of `Self` and accepts `&dyn
/// DynBackendStorage` instead of `&Self` for multi-operand operations.
///
/// For binary and multi-operand methods (`cmp`, `binary_op`, `where_cond`,
/// etc.), the implementation must downcast the `&dyn DynBackendStorage`
/// argument to its concrete type. Operations between different
/// backends are treated as a device mismatch.
pub trait DynBackendStorage: Send + Sync + std::fmt::Debug {
    /// Clone the elements described by `layout` into a new storage.
    fn try_clone_dyn(&self, layout: &Layout) -> Result<Box<dyn DynBackendStorage>>;

    /// The element dtype of this storage.
    fn dtype_dyn(&self) -> DType;

    /// The device that owns this storage.
    fn device_dyn(&self) -> &dyn DynBackendDevice;

    /// Return a cloned `Arc` handle to the owning device.
    fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice>;

    /// Copy the entire storage to a [`HostBuffer`](fuel_ir::HostBuffer).
    fn to_host_buffer_dyn(&self) -> Result<HostBuffer>;

    /// Deprecated alias for `to_host_buffer_dyn`.
    fn to_cpu_storage_dyn(&self) -> Result<HostBuffer> {
        self.to_host_buffer_dyn()
    }

    /// Elementwise affine: `x * mul + add`.
    fn affine_dyn(&self, layout: &Layout, mul: f64, add: f64)
    -> Result<Box<dyn DynBackendStorage>>;

    /// Raise every element to the power `e`.
    fn powf_dyn(&self, layout: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>>;

    /// ELU activation.
    fn elu_dyn(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>>;

    /// Reduction (sum, min, max, …) over `axes`.
    fn reduce_op_dyn(
        &self,
        op: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Elementwise comparison, returning a `U8` buffer.
    fn cmp_dyn(
        &self,
        op: CmpOp,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Cast to a different dtype.
    fn to_dtype_dyn(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>>;

    /// Elementwise unary operation.
    fn unary_op_dyn(&self, layout: &Layout, op: UnaryOp) -> Result<Box<dyn DynBackendStorage>>;

    /// Elementwise binary operation.
    fn binary_op_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Select `on_true` where condition is nonzero, `on_false` elsewhere.
    fn where_cond_dyn(
        &self,
        cond_layout: &Layout,
        on_true: &dyn DynBackendStorage,
        on_true_layout: &Layout,
        on_false: &dyn DynBackendStorage,
        on_false_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 1-D convolution.
    fn conv1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 1-D transposed convolution.
    fn conv_transpose1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D convolution.
    fn conv2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D transposed convolution.
    fn conv_transpose2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D average pooling.
    fn avg_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D max pooling.
    fn max_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 1-D nearest-neighbor upsampling.
    fn upsample_nearest1d_dyn(
        &self,
        layout: &Layout,
        target_size: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D nearest-neighbor upsampling.
    fn upsample_nearest2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// 2-D bilinear upsampling.
    fn upsample_bilinear2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Gather elements along `dim` using `ids`.
    fn gather_dyn(
        &self,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Scatter-set `src` into `self` along `dim` at positions in `ids`.
    fn scatter_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()>;

    /// Scatter-add `src` into `self` along `dim` at positions in `ids`.
    fn scatter_add_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()>;

    /// Select slices along `dim` at positions in `ids`.
    fn index_select_dyn(
        &self,
        ids: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Index-add: accumulate `src` slices into `self` along `dim` at `ids`.
    fn index_add_dyn(
        &self,
        self_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Matrix multiply.
    fn matmul_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Copy strided source into `dst` starting at `dst_offset`.
    fn copy_strided_src_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()>;

    /// Copy a 2-D tile from `self` into `dst`.
    fn copy2d_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()>;

    /// Set every element described by `layout` to `value`.
    fn const_set_dyn(&mut self, value: Scalar, layout: &Layout) -> Result<()>;

    /// Downcast to the concrete storage type.
    fn as_any(&self) -> &dyn Any;

    /// Downcast to the concrete storage type (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Object-safe device abstraction. The generic `BackendDevice` trait this
/// once mirrored was DELETED in the 2026-04-30 backend-agnostic refactor;
/// backends implement this trait directly.
///
/// All factory methods return `Box<dyn DynBackendStorage>` instead of
/// `Self::Storage`, and there is no `storage_from_slice` (use
/// [`storage_from_cpu_storage`](DynBackendDevice::storage_from_cpu_storage_dyn)
/// after constructing a [`HostBuffer`] on the host side).
pub trait DynBackendDevice: Send + Sync + std::fmt::Debug {
    /// Canonical device location.
    fn location_dyn(&self) -> DeviceLocation;

    /// Whether `self` and `other` refer to the same physical device.
    fn same_device_dyn(&self, other: &dyn DynBackendDevice) -> bool;

    /// Whether this device has native BF16 support.
    ///
    /// Defaults to `false`. Override in GPU backends that support BF16 natively.
    fn supports_bf16(&self) -> bool {
        false
    }

    /// Allocate zero-initialized storage.
    fn zeros_impl_dyn(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn DynBackendStorage>>;

    /// Allocate uninitialized storage.
    ///
    /// # Safety
    ///
    /// Caller must initialize all elements before reading.
    unsafe fn alloc_uninit_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Create storage from a host buffer (borrowed).
    fn storage_from_host_buffer_dyn(&self, buf: &HostBuffer) -> Result<Box<dyn DynBackendStorage>>;

    /// Create storage from a host buffer (owned).
    fn storage_from_host_buffer_owned_dyn(
        &self,
        buf: HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Deprecated alias for `storage_from_host_buffer_dyn`.
    fn storage_from_cpu_storage_dyn(&self, cpu: &HostBuffer) -> Result<Box<dyn DynBackendStorage>> {
        self.storage_from_host_buffer_dyn(cpu)
    }

    /// Deprecated alias for `storage_from_host_buffer_owned_dyn`.
    fn storage_from_cpu_storage_owned_dyn(
        &self,
        cpu: HostBuffer,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.storage_from_host_buffer_owned_dyn(cpu)
    }

    /// Fill storage with uniform random values in `[lo, hi)`.
    fn rand_uniform_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        hi: f64,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Fill storage with normally distributed random values.
    fn rand_normal_dyn(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn DynBackendStorage>>;

    /// Set the RNG seed.
    fn set_seed_dyn(&self, seed: u64) -> Result<()>;

    /// Get the current RNG seed.
    fn get_current_seed_dyn(&self) -> Result<u64>;

    /// Block until all pending operations complete.
    fn synchronize_dyn(&self) -> Result<()>;

    /// Downcast to the concrete device type.
    fn as_any(&self) -> &dyn Any;

    /// Optional quantized-kernel adapter for backends that support GGUF/ggml
    /// block formats. fuel-core walks this to allocate `qzeros` storage and
    /// load pre-quantized blocks without naming concrete backend types.
    /// Backends that don't support quantization return `None` (the default).
    fn as_quantized_kernels(&self) -> Option<&dyn QuantizedDeviceKernels> {
        None
    }

    /// Optional runtime-state adapter (backend contract v0.4, Tier 1).
    /// Backends whose device type implements
    /// [`crate::backend::BackendRuntime`] return `Some(self)` so
    /// holders of an `Arc<dyn DynBackendDevice>` (the picker arc's
    /// pressure-aware runtime selectors) can query live memory
    /// signals without naming concrete backend types. Backends
    /// without a runtime impl keep the default `None` — selectors
    /// treat that as [`fuel_ir::backend::FitStatus::Unknown`] (no
    /// signal), never as pressure.
    fn as_backend_runtime(&self) -> Option<&dyn crate::backend::BackendRuntime> {
        None
    }
}

// ===========================================================================
// Quantized kernel registration traits (merged here from the former `quantized` module
// so the two trait families no longer form a module cycle; `crate::quantized` re-exports them).
//
// Backend-agnostic quantized kernel registration traits.
//
// Per-backend quantized fast paths (CUDA, Metal, future Vulkan/CPU-SIMD)
// implement [`DynQuantizedStorage`] for their concrete storage type and
// [`QuantizedDeviceKernels`] for their device handle. fuel-core dispatches
// through these traits without naming concrete backend types.
//
// The CPU "fast path" is the bare ggml block math (avx/neon
// vec_dots inside `k_quants.rs`); it stays in fuel-core because
// `BlockQX` types and the file-format readers (gguf/ggml/imatrix) are a
// single cohesive numerics unit. The trait still covers the CPU case via
// a fuel-core-side adapter (see `fuel-core/src/quantized/mod.rs`).
//
// The `GgmlDType` block-format *data* tag lives in [`fuel_ir::quantized`].
// ===========================================================================

/// Object-safe per-backend quantized storage. Each backend (CPU, CUDA,
/// Metal, ...) supplies a concrete type that implements this trait;
/// fuel-core holds them as `Box<dyn DynQuantizedStorage>` and dispatches
/// without naming backends.
///
/// The `_src` arguments to `quantize*` are typed as `&dyn DynBackendStorage`
/// so the implementor can downcast to its own concrete storage; the
/// `_onto` variants take a `HostBuffer`-shaped CPU source.
pub trait DynQuantizedStorage: Send + Sync + std::fmt::Debug {
    fn dtype(&self) -> GgmlDType;
    fn block_size(&self) -> usize;
    fn storage_size_in_bytes(&self) -> usize;

    /// Quantize an in-device source storage onto self.
    fn quantize(&mut self, src: &dyn DynBackendStorage) -> Result<()>;

    /// Quantize with importance-matrix weighting.
    fn quantize_imatrix(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()>;

    /// Quantize a CPU source onto self (cross-device).
    fn quantize_onto(&mut self, src: &dyn DynBackendStorage) -> Result<()>;

    /// Quantize with importance matrix from a CPU source.
    fn quantize_imatrix_onto(
        &mut self,
        src: &dyn DynBackendStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()>;

    /// Dequantize to backend-native f32 storage.
    fn dequantize(&self, elem_count: usize) -> Result<Box<dyn DynBackendStorage>>;

    /// Dequantize to f16 (CUDA fast path; default impl rejects).
    fn dequantize_f16(&self, _elem_count: usize) -> Result<Box<dyn DynBackendStorage>> {
        Err(Error::Msg("dequantize_f16 not supported on this backend".into()).bt())
    }

    /// Raw bytes of the quantized data (host-readable copy).
    fn data(&self) -> Result<Cow<'_, [u8]>>;

    /// Device pointer for callers that need raw addressing (CUDA only today).
    fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::Msg("device_ptr not supported on this backend".into()).bt())
    }

    /// QMatMul forward against an in-device input storage. Returns a fresh
    /// device storage and its output shape. `self_shape` is the weight shape
    /// (the QTensor's own shape).
    fn fwd(
        &self,
        self_shape: &Shape,
        input: &dyn DynBackendStorage,
        layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)>;

    /// Indexed MoE forward (CUDA-only today; default rejects).
    fn indexed_moe_forward(
        &self,
        _self_shape: &Shape,
        _input: &dyn DynBackendStorage,
        _input_layout: &Layout,
        _ids: &dyn DynBackendStorage,
        _ids_layout: &Layout,
    ) -> Result<(Box<dyn DynBackendStorage>, Shape)> {
        Err(Error::Msg("indexed_moe_forward not supported on this backend".into()).bt())
    }

    /// Identity returns the host device tag this storage is on.
    fn as_any(&self) -> &dyn Any;

    /// Owning device handle. Lets fuel-core recover a `Device` from a
    /// `QTensor` without keeping a parallel `device` field on QTensor.
    fn device_arc_dyn(&self) -> std::sync::Arc<dyn DynBackendDevice>;
}

/// Backend device → quantized-kernel constructors. fuel-core looks this up
/// per Device to allocate fresh QStorage of a given dtype, or to load
/// pre-quantized bytes.
pub trait QuantizedDeviceKernels: DynBackendDevice {
    /// Allocate a zero-initialized quantized storage on this device.
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<Box<dyn DynQuantizedStorage>>;

    /// Load pre-quantized block-format bytes onto this device. `dtype`
    /// describes the block format; `data` is interpreted as a flat byte
    /// slice of those blocks.
    fn load_quantized(
        &self,
        dtype: GgmlDType,
        data: Cow<'_, [u8]>,
    ) -> Result<Box<dyn DynQuantizedStorage>>;
}
