// SPDX-License-Identifier: MIT OR Apache-2.0
//! TopKLastDim: the `k` largest values along the last axis and their indices
//! (ROADMAP row 172 item 2 step A; design:
//! `docs/design-notes/2026-11-25-row172-item2-matmul-topk-epilogue.md`).
//!
//! Multi-output: slot 0 = `values [.., k]` (the input dtype), slot 1 =
//! `indices [.., k]` (`U32`, as `ArgMaxDim`). Descending by value; NaN sorts as
//! the largest (`torch.topk`); ties take the lower index first. `k` is static
//! (it is part of the node's identity) and is validated at graph-build time.

use crate::registry::FusedOpParams;
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Shape};

/// Build-time validation of `k` against the reduced axis length `n`. A typed
/// error, never a panic: `1 <= k <= n`.
pub fn validate_k(k: usize, n: usize) -> Result<(), String> {
    if k == 0 {
        return Err("TopKLastDim: k must be >= 1".to_string());
    }
    if k > n {
        return Err(format!(
            "TopKLastDim: k = {k} exceeds the last axis length {n}"
        ));
    }
    Ok(())
}

fn k_of(params: &FusedOpParams) -> usize {
    match params {
        FusedOpParams::TopKLastDim { k } => *k,
        _ => 0,
    }
}

/// Slot-0 shape: the input shape with its last axis replaced by `k`.
pub fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    let mut dims = input_shapes[0].dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = k_of(params);
    }
    Shape::from_dims(&dims)
}

/// Dtype rule (slot 0): the input dtype.
pub fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    input_dtypes[0]
}

/// Multi-output authoring fn: `values` then `indices`.
pub fn output_views(
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    params: &FusedOpParams,
) -> Vec<OutputViewSpec> {
    let shape = shape_rule(input_shapes, params);
    vec![
        OutputViewSpec::contiguous(input_dtypes[0], shape.clone()),
        OutputViewSpec::contiguous(DType::U32, shape),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_is_validated_at_build_time() {
        assert!(validate_k(1, 8).is_ok());
        assert!(validate_k(8, 8).is_ok());
        assert!(validate_k(0, 8).is_err(), "k = 0 declined");
        assert!(validate_k(9, 8).is_err(), "k > N declined");
    }

    #[test]
    fn slot_shapes_replace_the_last_axis_with_k() {
        let p = FusedOpParams::TopKLastDim { k: 3 };
        let ins = [Shape::from_dims(&[2, 5, 8])];
        assert_eq!(shape_rule(&ins, &p).dims(), &[2, 5, 3]);
        let views = output_views(&ins, &[DType::F32], &p);
        assert_eq!(views.len(), 2, "values and indices");
        assert_eq!(
            (views[0].dtype, views[0].shape.dims()),
            (DType::F32, &[2usize, 5, 3][..])
        );
        assert_eq!(
            (views[1].dtype, views[1].shape.dims()),
            (DType::U32, &[2usize, 5, 3][..])
        );
        // The multi-output invariant: slot 0 equals shape_rule / dtype_rule.
        assert_eq!(views[0].shape.dims(), shape_rule(&ins, &p).dims());
        assert_eq!(views[0].dtype, dtype_rule(&[DType::F32], &p));
    }

    /// Sabotage sibling: a different k moves the shapes (the rule reads k).
    #[test]
    fn a_different_k_moves_the_shape() {
        let ins = [Shape::from_dims(&[4, 8])];
        let a = shape_rule(&ins, &FusedOpParams::TopKLastDim { k: 2 });
        let b = shape_rule(&ins, &FusedOpParams::TopKLastDim { k: 5 });
        assert_ne!(a.dims(), b.dims());
    }
}
