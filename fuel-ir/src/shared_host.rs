// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host data that is shared (an `Arc<[T]>`), not owned (a `Vec<T>`).
//!
//! Row 173 (d): a graph `Const` built from a model's weight `Arc` used to copy the
//! weights (`Arc<[T]>::to_vec`). A backend that can keep the `Arc` instead receives
//! it as a [`SharedHostData`]; every other backend takes the default, which is the old
//! copy. `HostBuffer` itself is unchanged.

use crate::{DType, HostBuffer};
use half::{bf16, f16};
use std::sync::Arc;

/// A dtype-tagged shared host slice. Cloning clones the `Arc`, never the data.
#[derive(Debug, Clone)]
pub enum SharedHostData {
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
    BF16(Arc<[bf16]>),
    F16(Arc<[f16]>),
    U32(Arc<[u32]>),
}

impl SharedHostData {
    /// Element dtype.
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::U32(_) => DType::U32,
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    /// True when there are no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of the data in bytes.
    pub fn size_in_bytes(&self) -> usize {
        self.len() * self.dtype().size_in_bytes()
    }

    /// A deep copy as an owned [`HostBuffer`] (the pre-row-173 behaviour).
    pub fn to_host_buffer(&self) -> HostBuffer {
        match self {
            Self::F32(v) => HostBuffer::F32(v.to_vec()),
            Self::F64(v) => HostBuffer::F64(v.to_vec()),
            Self::BF16(v) => HostBuffer::BF16(v.to_vec()),
            Self::F16(v) => HostBuffer::F16(v.to_vec()),
            Self::U32(v) => HostBuffer::U32(v.to_vec()),
        }
    }

    /// The address of the first element, for `ptr_eq`-style sharing checks.
    pub fn data_ptr(&self) -> *const u8 {
        match self {
            Self::F32(v) => v.as_ptr().cast(),
            Self::F64(v) => v.as_ptr().cast(),
            Self::BF16(v) => v.as_ptr().cast(),
            Self::F16(v) => v.as_ptr().cast(),
            Self::U32(v) => v.as_ptr().cast(),
        }
    }
}
