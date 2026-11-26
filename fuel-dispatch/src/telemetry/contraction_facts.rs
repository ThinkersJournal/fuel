// SPDX-License-Identifier: MIT OR Apache-2.0
//! The contraction facts `Op::MatMul` declares, built from its live operands
//! (union row 172, first slice).
//!
//! `Op::MatMul` is a unit variant whose semantics fix its roles: `[.., M, K] x
//! [.., K, N] -> [.., M, N]`, every leading axis a batch axis. This module is
//! the **op declaring its roles** (the lane recommendation the PM accepted for
//! row 172): the planner reads them here, it does not infer them from layouts.
//! A fused op will declare its roles explicitly in `fuel-ir`/`fuel-graph`
//! later; none of that is needed for `MatMul`.
//!
//! Before this module no production path built a [`GemCell`]
//! (`derive_structure_key_token` had no caller outside tests).

use fuel_ir::DType;

use super::structure_key::FdxOperandDesc;
use super::structure_key_derive::{
    CellOperand, DeclaredFacts, DeriveDecline, GemCell, GemMathPrecision,
    derive_structure_key_token_typed,
};

/// The accumulator dtype the contraction kernels use for `dt`, or `None` when
/// Fuel has no declared accumulator for it (no key then, never a guess).
fn accumulator_for(dt: DType) -> Option<DType> {
    match dt {
        DType::F32 | DType::F16 | DType::BF16 => Some(DType::F32),
        DType::F64 => Some(DType::F64),
        _ => None,
    }
}

/// The contraction facts of a `MatMul` over `a`, `b` producing `out`. `None`
/// when the operands do not form a plain batched matmul (rank below 2, unequal
/// ranks, mismatched K, mixed dtypes or an undeclared accumulator).
pub fn matmul_declared_facts(
    a: &FdxOperandDesc,
    b: &FdxOperandDesc,
    out: &FdxOperandDesc,
) -> Option<DeclaredFacts> {
    let r = a.shape.len();
    if r < 2 || b.shape.len() != r || out.shape.len() != r {
        return None;
    }
    if a.dtype != b.dtype || a.dtype != out.dtype {
        return None;
    }
    let (m, k) = (a.shape[r - 2], a.shape[r - 1]);
    let (k2, n) = (b.shape[r - 2], b.shape[r - 1]);
    if k != k2 || out.shape[r - 2] != m || out.shape[r - 1] != n {
        return None;
    }
    let batch = if r > 2 {
        Some(out.shape[..r - 2].iter().product::<i64>())
    } else {
        None
    };
    Some(DeclaredFacts::contraction(GemCell {
        m,
        n,
        k,
        batch,
        weight_dtype: b.dtype,
        acc_dtype: accumulator_for(a.dtype)?,
        out_dtype: out.dtype,
        math_precision: GemMathPrecision::BitStable,
        scale_dtype: None,
        block_length: None,
    }))
}

/// Fuel's own `sk8` token for a `MatMul` over `a`, `b` -> `out` on `target`
/// (a KISS target string such as `cuda:sm89`). `Ok(None)` when the node is not
/// a plain matmul; `Err` carries the derivation's typed decline.
pub fn matmul_structure_key(
    a: &FdxOperandDesc,
    b: &FdxOperandDesc,
    out: &FdxOperandDesc,
    target: &str,
) -> Result<Option<String>, DeriveDecline> {
    let Some(facts) = matmul_declared_facts(a, b, out) else {
        return Ok(None);
    };
    // KISS-OPS-6.17-0002: a kernel that flushes subnormals must decline `Exact`, and the cell declares
    // `GemMathPrecision::BitStable`. Metal flushes f32 and bf16 subnormal operands and results (measured on the hosted
    // macOS device: vulkane run 45748530618, fuel run 45748920513), so no key is derived for them on a Metal target:
    // the contraction is declined, not mis-declared. f16 preserves subnormals (measured) and keeps its key.
    if target.starts_with("metal") && matches!(a.dtype, DType::F32 | DType::BF16) {
        return Ok(None);
    }
    let operands = [
        CellOperand::input(a.clone()),
        CellOperand::input(b.clone()),
        CellOperand::output(out.clone()),
    ];
    derive_structure_key_token_typed(&facts, &operands, target, None).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::{Layout, Shape};

    fn d(dims: &[usize], dt: DType) -> FdxOperandDesc {
        FdxOperandDesc::from_layout(&Layout::contiguous(Shape::from_dims(dims)), dt)
    }

    #[test]
    fn plain_and_batched_matmul_declare_their_roles() {
        let f = matmul_declared_facts(
            &d(&[8, 64], DType::F32),
            &d(&[64, 16], DType::F32),
            &d(&[8, 16], DType::F32),
        )
        .expect("plain");
        let c = f.contraction.expect("ctr");
        assert_eq!((c.m, c.n, c.k, c.batch), (8, 16, 64, None));
        let f = matmul_declared_facts(
            &d(&[2, 3, 8, 64], DType::F16),
            &d(&[2, 3, 64, 16], DType::F16),
            &d(&[2, 3, 8, 16], DType::F16),
        )
        .expect("batched");
        let c = f.contraction.expect("ctr");
        assert_eq!((c.m, c.n, c.k, c.batch), (8, 16, 64, Some(6)));
        assert_eq!(c.acc_dtype, DType::F32);
    }

    // KISS-OPS 6.17-0002: a kernel that flushes subnormals (FTZ/DAZ) is non-conformant under bit-stable and MUST decline
    // `Exact`. Metal flushes f32 and bf16 subnormals (operands and results) and preserves f16 (vulkane's hosted-macOS
    // probe, run 45748530618; Fuel's own record `subnormal_behaviour_record` in fuel-metal-kernels). The declared `st`
    // math-precision of a contraction cell must therefore not be derived for a Metal target at f32/bf16.
    #[test]
    fn a_metal_target_declines_the_bit_stable_key_for_f32_and_bf16() {
        for dt in [DType::F32, DType::BF16] {
            let (a, b, o) = (d(&[8, 64], dt), d(&[64, 16], dt), d(&[8, 16], dt));
            assert_eq!(
                matmul_structure_key(&a, &b, &o, "metal:apple9").map(|k| k.is_none()),
                Ok(true),
                "{dt:?} on a Metal target must decline (FTZ), not derive a bit-stable key"
            );
        }
    }

    /// Sabotage sibling: the same operands on a CUDA target still derive a key (the decline is the Metal flush, not a
    /// blanket decline), so the test above cannot pass by declining everything.
    #[test]
    fn a_cuda_target_still_derives_the_key_for_f32_and_bf16() {
        for dt in [DType::F32, DType::BF16] {
            let (a, b, o) = (d(&[8, 64], dt), d(&[64, 16], dt), d(&[8, 16], dt));
            let k = matmul_structure_key(&a, &b, &o, "cuda:sm89");
            assert!(matches!(k, Ok(Some(_))), "{dt:?} on cuda:sm89: {k:?}");
        }
    }

    #[test]
    fn a_non_matmul_shape_declares_nothing() {
        // K mismatch, rank 1, mixed dtype, integer accumulator: all decline.
        let a = d(&[8, 64], DType::F32);
        assert!(
            matmul_declared_facts(&a, &d(&[65, 16], DType::F32), &d(&[8, 16], DType::F32))
                .is_none()
        );
        assert!(
            matmul_declared_facts(
                &d(&[64], DType::F32),
                &d(&[64], DType::F32),
                &d(&[1], DType::F32)
            )
            .is_none()
        );
        assert!(
            matmul_declared_facts(&a, &d(&[64, 16], DType::F16), &d(&[8, 16], DType::F32))
                .is_none()
        );
        assert!(
            matmul_declared_facts(
                &d(&[8, 64], DType::U8),
                &d(&[64, 16], DType::U8),
                &d(&[8, 16], DType::U8)
            )
            .is_none()
        );
    }

    #[test]
    fn the_key_is_a_ctr_cell() {
        let t = matmul_structure_key(
            &d(&[8, 64], DType::F32),
            &d(&[64, 16], DType::F32),
            &d(&[8, 16], DType::F32),
            "cuda:sm89",
        )
        .expect("derives")
        .expect("a matmul");
        assert!(t.starts_with("sk8|ctr|f32|cuda:sm89|"), "{t}");
    }
}
