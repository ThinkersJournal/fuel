// SPDX-License-Identifier: MIT OR Apache-2.0
//! TopKLastDim on CUDA via baracuda's standalone block-bitonic `topk`
//! (`baracuda_kernels_topk_{f32,f64}_run`; ROADMAP row 172 item 2 step A).
//!
//! Caps (baracuda `TopkPlan`): `f32` / `f64` only, `row_len <= 1024`, `k <= 64`.
//! Outside them [`topk_last_dim`] returns a typed `Err` BEFORE any allocation or
//! launch; there is no per-shape decline to the CPU path in the binding table
//! (a binding is admitted by dtype/layout/rank, not by value).
//!
//! Output: Fuel's bundled `[values (rows*k of T) ; pad to 4 ; indices (rows*k of
//! u32)]` in ONE buffer, the layout `fuel_ir::storage::compose_bundle` computes;
//! baracuda's `i32` indices share bits with `u32` for the non-negative range.

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::status::check;

/// baracuda `SORT_MAX_ROW` and `TOPK_MAX_K` (the trailblazer caps).
pub const TOPK_MAX_ROW_LEN: usize = 1024;
pub const TOPK_MAX_K: usize = 64;

type TopkRun = unsafe extern "C" fn(
    batch: i32,
    row_len: i32,
    k: i32,
    largest: i32,
    x: *const std::ffi::c_void,
    y_vals: *mut std::ffi::c_void,
    y_idx: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Byte size of the bundled output and the 4-aligned offset of the indices slot.
pub fn bundle_layout(elem_size: usize, rows: usize, k: usize) -> (usize, usize) {
    let values_bytes = rows * k * elem_size;
    let idx_off = values_bytes.div_ceil(4) * 4;
    (idx_off + rows * k * 4, idx_off)
}

/// Validate the shape against baracuda's caps (pure; unit-testable off-GPU).
pub fn check_caps(rows: usize, n: usize, k: usize) -> Result<()> {
    if k == 0 || k > n {
        return Err(Error::Msg(format!("top_k_last_dim: k = {k} must be in 1..={n}")).bt());
    }
    if n > TOPK_MAX_ROW_LEN {
        return Err(Error::Msg(format!(
            "top_k_last_dim (CUDA/baracuda): last axis {n} exceeds the block-bitonic cap {TOPK_MAX_ROW_LEN}; \
             this shape runs on the CPU kernel only"
        ))
        .bt());
    }
    if k > TOPK_MAX_K {
        return Err(Error::Msg(format!(
            "top_k_last_dim (CUDA/baracuda): k = {k} exceeds the cap {TOPK_MAX_K}; \
             this shape runs on the CPU kernel only"
        ))
        .bt());
    }
    if i32::try_from(rows).is_err() {
        return Err(Error::Msg(format!("top_k_last_dim: {rows} rows overflow i32")).bt());
    }
    Ok(())
}

fn topk_last_dim(
    input: &CudaStorageBytes,
    rows: usize,
    n: usize,
    k: usize,
    elem_size: usize,
    kernel: TopkRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    check_caps(rows, n, k)?;
    let device = input.device().clone();
    let (total, idx_off) = bundle_layout(elem_size, rows, k);
    if rows == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out = device.alloc_zeros::<u8>(total)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let in_ptr = input.buffer().as_raw().0 as *const std::ffi::c_void;
    let base = out.as_raw().0;
    let vals_ptr = base as *mut std::ffi::c_void;
    let idx_ptr = (base + idx_off as u64) as *mut std::ffi::c_void;
    // SAFETY: the pointers address the live input and the freshly allocated
    // bundle (values at 0, indices at the 4-aligned `idx_off`); workspace is
    // null+0 (baracuda's topk needs none).
    let status = unsafe {
        kernel(
            rows as i32,
            n as i32,
            k as i32,
            1, // largest
            in_ptr,
            vals_ptr,
            idx_ptr,
            std::ptr::null_mut(),
            0,
            stream,
        )
    };
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(
        std::sync::Arc::new(out),
        device,
        total,
    ))
}

/// f32 TopK (`rows` rows of `n`, keep `k`), bundled output.
pub fn top_k_last_dim_f32(
    input: &CudaStorageBytes,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    topk_last_dim(
        input,
        rows,
        n,
        k,
        4,
        sys::baracuda_kernels_topk_f32_run,
        "topk_f32",
    )
}

/// f64 TopK, bundled output.
pub fn top_k_last_dim_f64(
    input: &CudaStorageBytes,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    topk_last_dim(
        input,
        rows,
        n,
        k,
        8,
        sys::baracuda_kernels_topk_f64_run,
        "topk_f64",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundle_layout_matches_compose_bundle() {
        // f32: 6 values (24 B), indices at 24; bf16-like odd case via elem 2.
        assert_eq!(bundle_layout(4, 2, 3), (24 + 24, 24));
        assert_eq!(bundle_layout(2, 1, 3), (8 + 12, 8), "6 B pads to 8");
        assert_eq!(bundle_layout(8, 2, 2), (32 + 16, 32));
    }

    #[test]
    fn caps_decline_before_any_launch() {
        assert!(check_caps(4, 1024, 64).is_ok());
        assert!(check_caps(4, 1025, 3).is_err(), "row_len over the cap");
        assert!(check_caps(4, 100, 65).is_err(), "k over the cap");
        assert!(check_caps(4, 8, 0).is_err() && check_caps(4, 8, 9).is_err());
    }
}
