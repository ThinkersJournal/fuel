// SPDX-License-Identifier: MIT OR Apache-2.0
//! Backend-agnostic quantized kernel registration traits.
//!
//! The traits live in [`crate::dyn_backend`] (they and the object-safe storage/device traits name
//! each other, so one module holds all of them); this module keeps the original public paths
//! (`fuel_backend_contract::quantized::{DynQuantizedStorage, QuantizedDeviceKernels}`).
//! The `GgmlDType` block-format *data* tag lives in [`fuel_ir::quantized`].

pub use crate::dyn_backend::{DynQuantizedStorage, QuantizedDeviceKernels};
