// SPDX-License-Identifier: MIT OR Apache-2.0
use fuel_ir::{DType, Layout};

/// Errors from the CUDA backend — driver, NVRTC, cuBLAS, and curand
/// failures, plus Fuel-local variants for missing kernels / dtype
/// mismatches / non-contiguous matmul.
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] baracuda_driver::Error),

    #[error(transparent)]
    Compiler(#[from] baracuda_nvrtc::Error),

    #[error(transparent)]
    Cublas(#[from] baracuda_cublas::Error),

    #[error(transparent)]
    Curand(#[from] baracuda_curand::Error),

    #[cfg(feature = "cudnn")]
    #[error(transparent)]
    Cudnn(#[from] baracuda_cudnn::Error),

    #[cfg(feature = "cudnn")]
    #[error(transparent)]
    CudnnLoader(#[from] baracuda_core::LoaderError),

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("unsupported dtype {dtype:?} for {op}")]
    UnsupportedDtype { dtype: DType, op: &'static str },

    /// No CUDA storage exists for `dtype`, carrying the TYPED reason from the
    /// single authority (`cuda_storage_status`) rather than erasing it into the
    /// reasonless `UnsupportedDtype`. GAP-161: the reason (unwired-yet vs
    /// encoding-unauthored vs impossible) is load-bearing and must survive to
    /// the caller, not live only in a source comment.
    #[error("no CUDA storage for {dtype:?} ({op}): {reason}")]
    StorageUnavailable {
        dtype: DType,
        op: &'static str,
        reason: &'static str,
    },

    #[error(
        "cannot view {byte_len} bytes as {dtype_name}: byte length not divisible by size_of::<{dtype_name}>() = {dtype_size}"
    )]
    InvalidDtypeBoundary {
        byte_len: usize,
        dtype_size: usize,
        dtype_name: &'static str,
    },

    #[error("baracuda-kernels {op}: status {code} ({reason})")]
    BaracudaKernel {
        op: &'static str,
        code: i32,
        reason: &'static str,
    },

    #[error("baracuda-kernels {op}: shape dim {dim_index} = {dim_value} exceeds i32 range")]
    BaracudaShapeOverflow {
        op: &'static str,
        dim_index: usize,
        dim_value: usize,
    },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error(
        "matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}"
    )]
    MatMulNonContiguous {
        lhs_stride: Box<Layout>,
        rhs_stride: Box<Layout>,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: baracuda_driver::Error,
        module_name: String,
    },
}

impl From<CudaError> for fuel_ir::Error {
    fn from(val: CudaError) -> Self {
        fuel_ir::Error::cuda(val)
    }
}

pub trait WrapErr<O> {
    fn w(self) -> std::result::Result<O, fuel_ir::Error>;
}

impl<O, E: Into<CudaError>> WrapErr<O> for std::result::Result<O, E> {
    fn w(self) -> std::result::Result<O, fuel_ir::Error> {
        self.map_err(|e| fuel_ir::Error::cuda(e.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `MatMulNonContiguous` carries two `Layout`s; held inline they made EVERY `Result<_, CudaError>` on the CUDA path
    /// 424 bytes (row 38, `clippy::large_enum_variant`). Born red against the inline layouts; the bound keeps the lint
    /// from coming back if a large payload is added inline again.
    #[test]
    fn cuda_error_stays_small() {
        let size = std::mem::size_of::<CudaError>();
        assert!(
            size <= 100,
            "CudaError is {size} bytes; box the large variant's payload"
        );
    }

    /// Boxing the layouts must not change what the error says: the Display text is pinned byte for byte.
    #[test]
    fn matmul_non_contiguous_display_is_unchanged() {
        let lhs = Layout::contiguous((2usize, 3usize));
        let rhs = Layout::contiguous((3usize, 4usize));
        let e = CudaError::MatMulNonContiguous {
            lhs_stride: Box::new(lhs),
            rhs_stride: Box::new(rhs),
            mnk: (2, 4, 3),
        };
        assert_eq!(
            e.to_string(),
            "matmul is only supported for contiguous tensors lstride: Layout { shape: [2, 3], stride: [3, 1], start_offset: 0 } rstride: Layout { shape: [3, 4], stride: [4, 1], start_offset: 0 } mnk: (2, 4, 3)"
        );
    }
}
