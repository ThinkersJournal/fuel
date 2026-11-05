// SPDX-License-Identifier: MIT OR Apache-2.0
//! TEST-ONLY CUDA binary wrappers, so `fuel-dispatch`'s own live-device tests can launch a real CUDA add/mul
//! without naming `fuel_cuda_dispatch::baracuda_dispatch` (which sits above this crate since ROADMAP row 137
//! step 6.2). They forward to the same `fuel_cuda_backend::baracuda::binary` drivers as the production
//! wrappers and are NOT registered anywhere.
//!
//! Temporary: the tests that use them (`jit_ingest`'s kiss-advisory / verify-candidate fixtures and
//! `CudaInvoker`'s end-to-end add) move to `fuel-cuda-dispatch` with the JIT modules in step 6.3, and this
//! file goes with them.

use crate::dispatch::{cuda_input, cuda_output, read_storage, write_storage};
use crate::kernel::OpParams;
use fuel_ir::{Error, Layout, Result};
use fuel_memory::Storage;
use std::sync::{Arc, RwLock};

macro_rules! binary_wrapper {
    ($name:ident, $driver:path) => {
        pub(crate) fn $name(
            inputs: &[Arc<RwLock<Storage>>],
            outputs: &mut [Arc<RwLock<Storage>>],
            layouts: &[Layout],
            _params: &OpParams,
        ) -> Result<()> {
            if inputs.len() != 2 || outputs.len() != 1 {
                return Err(Error::Msg(format!(
                    concat!(
                        stringify!($name),
                        ": expected 2 inputs + 1 output, got {} + {}"
                    ),
                    inputs.len(),
                    outputs.len(),
                ))
                .bt());
            }
            let lhs = read_storage(&inputs[0])?;
            let rhs = read_storage(&inputs[1])?;
            let mut out = write_storage(&outputs[0])?;
            $driver(
                cuda_input(&lhs)?,
                cuda_input(&rhs)?,
                layouts.first(),
                layouts.get(1),
                cuda_output(&mut out)?,
            )
        }
    };
}

binary_wrapper!(
    add_f32,
    fuel_cuda_backend::baracuda::binary::binary_add_f32_into
);
binary_wrapper!(
    add_f64,
    fuel_cuda_backend::baracuda::binary::binary_add_f64_into
);
binary_wrapper!(
    mul_f32,
    fuel_cuda_backend::baracuda::binary::binary_mul_f32_into
);

/// A `BindingEntry` for tests whose runner ignores the entry (the retained "the measurement sees a
/// perturbed output" siblings in `fkc/verify/accuracy_*`): they only need SOME entry value to hand to the
/// measurement. That the real registered entries exist is pinned by the CUDA binding census in
/// `fuel-cuda-dispatch/tests/binding_census.rs`.
pub(crate) fn stub_entry() -> crate::kernel::BindingEntry {
    crate::kernel::BindingEntry {
        kernel: add_f32,
        caps: crate::kernel::KernelCaps::empty(),
        precision: crate::fused::PrecisionGuarantee::UNAUDITED,
        cost: crate::kernel::unknown_cost,
        kernel_source: "test-stub",
        is_generic: false,
        kernel_revision_hash: 0,
        cost_expr: None,
    }
}
