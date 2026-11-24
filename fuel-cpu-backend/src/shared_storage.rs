// SPDX-License-Identifier: MIT OR Apache-2.0
//! `SharedHostStorage`: CPU storage that keeps a model's `Arc<[T]>` instead of copying it.
//!
//! Row 173 (d), option C. `NodeHandle::from_f32` and its siblings used to deep-copy the
//! weight `Arc` into a private `Vec` for every `Const` (1,508 MB per Qwen3-0.6B prefill).
//! This storage holds the `Arc` itself.
//!
//! # The two states
//!
//! * **Shared** (`shared` is `Some`): reads are served from the `Arc`; nothing was copied.
//!   A compute op that needs a `&HostBuffer` view (every CPU kernel does) materialises one
//!   owned copy, ONCE, into `cached`, which is exactly the copy the old build made up
//!   front, so a CPU realize costs what it always did, and a realize that never reads the
//!   slot (the warm residency path) pays nothing.
//! * **Owned** (`shared` is `None`, `cached` holds the data): the buffer was written.
//!
//! # Copy-on-write
//!
//! The shared `Arc` is the model's own weights and must never change. Every entry point
//! that can write goes through [`SharedHostStorage::make_owned`]: `copy_strided_src_dyn` and
//! `copy2d_dyn` (the destination is reached by `downcast_mut`, i.e. `as_any_mut`),
//! `const_set_dyn`, `scatter_set_dyn`, `scatter_add_set_dyn` and `as_any_mut` itself. After
//! `make_owned` the storage no longer references the `Arc`.

use crate::dyn_impl::{CpuBackendDevice, CpuStorage};
use fuel_backend_contract::dyn_backend::{DynBackendDevice, DynBackendStorage};
use fuel_ir::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use fuel_ir::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use fuel_ir::{DType, HostBuffer, Layout, Result, Scalar, SharedHostData};
use std::any::Any;
use std::sync::{Arc, OnceLock};

/// See the module docs.
#[derive(Debug, Clone)]
pub struct SharedHostStorage {
    shared: Option<SharedHostData>,
    /// The materialised owned copy: a cache while `shared` is `Some`, the data once it is `None`.
    cached: OnceLock<CpuStorage>,
}

impl SharedHostStorage {
    /// Wrap shared data without copying it.
    pub fn new(data: SharedHostData) -> Self {
        Self {
            shared: Some(data),
            cached: OnceLock::new(),
        }
    }

    /// The shared data, while this storage still references it (`None` after a write).
    pub fn shared(&self) -> Option<&SharedHostData> {
        self.shared.as_ref()
    }

    /// True while the storage still aliases the source `Arc`.
    pub fn is_shared(&self) -> bool {
        self.shared.is_some()
    }

    /// True once a CPU compute op has materialised the owned copy.
    pub fn is_materialised(&self) -> bool {
        self.cached.get().is_some()
    }

    /// The `&CpuStorage` view every compute op needs; copies on first use only.
    fn view(&self) -> &CpuStorage {
        self.cached.get_or_init(|| {
            let data = self
                .shared
                .as_ref()
                .expect("SharedHostStorage invariant: `cached` is set whenever `shared` is None");
            CpuStorage(data.to_host_buffer())
        })
    }

    /// Copy-on-write: take private ownership of the data, drop the `Arc`.
    fn make_owned(&mut self) -> &mut CpuStorage {
        self.view();
        self.shared = None;
        self.cached
            .get_mut()
            .expect("view() just initialised `cached`")
    }
}

impl DynBackendStorage for SharedHostStorage {
    fn try_clone_dyn(&self, _layout: &Layout) -> Result<Box<dyn DynBackendStorage>> {
        // A clone of shared data is another reference to it: copy-on-write keeps that safe.
        Ok(Box::new(self.clone()))
    }

    fn dtype_dyn(&self) -> DType {
        match &self.shared {
            Some(d) => d.dtype(),
            None => self.view().0.dtype(),
        }
    }

    fn device_dyn(&self) -> &dyn DynBackendDevice {
        // Stateless device: do not materialise a copy just to answer this.
        &CpuBackendDevice
    }

    fn device_arc_dyn(&self) -> Arc<dyn DynBackendDevice> {
        Arc::new(CpuBackendDevice)
    }

    fn to_host_buffer_dyn(&self) -> Result<HostBuffer> {
        match &self.shared {
            // The residency-OFF / cold-upload read: one copy, as before, and none cached.
            Some(d) => Ok(d.to_host_buffer()),
            None => self.view().to_host_buffer_dyn(),
        }
    }

    fn const_set_dyn(&mut self, value: Scalar, layout: &Layout) -> Result<()> {
        self.make_owned().const_set_dyn(value, layout)
    }

    fn as_any(&self) -> &dyn Any {
        // Callers downcast to `CpuStorage` to read; hand them the materialised view.
        self.view()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        // A `&mut` escape hatch can write: copy first.
        self.make_owned()
    }

    fn affine_dyn(
        &self,
        layout: &Layout,
        mul: f64,
        add: f64,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().affine_dyn(layout, mul, add)
    }

    fn powf_dyn(&self, layout: &Layout, e: f64) -> Result<Box<dyn DynBackendStorage>> {
        self.view().powf_dyn(layout, e)
    }

    fn elu_dyn(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn DynBackendStorage>> {
        self.view().elu_dyn(layout, alpha)
    }

    fn reduce_op_dyn(
        &self,
        op: ReduceOp,
        layout: &Layout,
        axes: &[usize],
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().reduce_op_dyn(op, layout, axes)
    }

    fn cmp_dyn(
        &self,
        op: CmpOp,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().cmp_dyn(op, rhs, lhs_layout, rhs_layout)
    }

    fn to_dtype_dyn(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn DynBackendStorage>> {
        self.view().to_dtype_dyn(layout, dtype)
    }

    fn unary_op_dyn(&self, layout: &Layout, op: UnaryOp) -> Result<Box<dyn DynBackendStorage>> {
        self.view().unary_op_dyn(layout, op)
    }

    fn binary_op_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().binary_op_dyn(rhs, lhs_layout, rhs_layout, op)
    }

    fn where_cond_dyn(
        &self,
        cond_layout: &Layout,
        on_true: &dyn DynBackendStorage,
        on_true_layout: &Layout,
        on_false: &dyn DynBackendStorage,
        on_false_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().where_cond_dyn(
            cond_layout,
            on_true,
            on_true_layout,
            on_false,
            on_false_layout,
        )
    }

    fn conv1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().conv1d_dyn(l, kernel, kernel_l, params)
    }

    fn conv_transpose1d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view()
            .conv_transpose1d_dyn(l, kernel, kernel_l, params)
    }

    fn conv2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().conv2d_dyn(l, kernel, kernel_l, params)
    }

    fn conv_transpose2d_dyn(
        &self,
        l: &Layout,
        kernel: &dyn DynBackendStorage,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view()
            .conv_transpose2d_dyn(l, kernel, kernel_l, params)
    }

    fn avg_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().avg_pool2d_dyn(layout, kernel, stride)
    }

    fn max_pool2d_dyn(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().max_pool2d_dyn(layout, kernel, stride)
    }

    fn upsample_nearest1d_dyn(
        &self,
        layout: &Layout,
        target_size: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().upsample_nearest1d_dyn(layout, target_size)
    }

    fn upsample_nearest2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view()
            .upsample_nearest2d_dyn(layout, target_h, target_w)
    }

    fn upsample_bilinear2d_dyn(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().upsample_bilinear2d_dyn(
            layout,
            target_h,
            target_w,
            align_corners,
            scale_h,
            scale_w,
        )
    }

    fn gather_dyn(
        &self,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().gather_dyn(src_layout, ids, ids_layout, dim)
    }

    fn scatter_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.make_owned()
            .scatter_set_dyn(self_layout, src, src_layout, ids, ids_layout, dim)
    }

    fn scatter_add_set_dyn(
        &mut self,
        self_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.make_owned()
            .scatter_add_set_dyn(self_layout, src, src_layout, ids, ids_layout, dim)
    }

    fn index_select_dyn(
        &self,
        ids: &dyn DynBackendStorage,
        src_layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view()
            .index_select_dyn(ids, src_layout, ids_layout, dim)
    }

    fn index_add_dyn(
        &self,
        self_layout: &Layout,
        ids: &dyn DynBackendStorage,
        ids_layout: &Layout,
        src: &dyn DynBackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view()
            .index_add_dyn(self_layout, ids, ids_layout, src, src_layout, dim)
    }

    fn matmul_dyn(
        &self,
        rhs: &dyn DynBackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn DynBackendStorage>> {
        self.view().matmul_dyn(rhs, bmnk, lhs_layout, rhs_layout)
    }

    fn copy_strided_src_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        self.view()
            .copy_strided_src_dyn(dst, dst_offset, src_layout)
    }

    fn copy2d_dyn(
        &self,
        dst: &mut dyn DynBackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        self.view().copy2d_dyn(
            dst,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
    }
}
