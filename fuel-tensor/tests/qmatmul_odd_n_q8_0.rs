// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 170: a quantized weight whose byte count is not a multiple of 4 (an odd `N` of Q8_0, 34-byte
//! blocks) is stored rounded UP to whole `u32` elements. `Tensor::qmatmul` must accept that length and
//! still reject a buffer too short to hold the blocks.
//!
//! Reported by lightbulb (2026-11-22): `output.weight` with K=32, N=97, Q8_0 failed at serve time with
//! `weight_bytes has 825 u32 elements, expected 824`. The check divided the byte count by 4 and truncated.

use fuel_graph::QuantType;
use fuel_ir::Shape;
use fuel_tensor::Device;
use fuel_tensor::lazy::Tensor;

/// Build the `qmatmul` node for `n` rows of K=32 Q8_0 with a weight buffer of `u32_elems` elements.
fn build(n: usize, u32_elems: usize) -> Result<Tensor, fuel_ir::Error> {
    let dev = Device::cpu();
    let k = 32;
    let x = Tensor::from_f32(vec![0.5; k], Shape::from_dims(&[1, k]), &dev).unwrap();
    let w = x
        .const_u32_like(vec![0u32; u32_elems], Shape::from_dims(&[u32_elems]))
        .unwrap();
    x.qmatmul(&w, QuantType::Q8_0, k, n)
}

#[test]
fn odd_n_q8_0_weight_rounded_up_to_whole_u32_is_accepted() {
    // 97 rows * 34 bytes = 3298 bytes = 824.5 u32 -> 825 elements.
    let bytes = 97 * QuantType::Q8_0.bytes_per_block();
    assert_eq!(
        bytes % 4,
        2,
        "the case under test: a byte count that is not a multiple of 4"
    );
    let y = build(97, bytes.div_ceil(4)).unwrap_or_else(|e| panic!("odd-N Q8_0 rejected: {e:?}"));
    assert_eq!(y.shape().dims(), &[1, 97]);
}

#[test]
fn even_n_q8_0_control_is_accepted() {
    // 98 * 34 = 3332 bytes = 833 u32 exactly: the control that always passed.
    let y = build(98, 98 * 34 / 4).unwrap();
    assert_eq!(y.shape().dims(), &[1, 98]);
}

/// Sabotage sibling: the fix must not disarm the check. 824 u32 hold 3296 bytes, two short of the
/// 3298 the blocks need; a wrong length of any other kind must also still be refused.
#[test]
fn a_weight_buffer_too_short_or_too_long_is_still_rejected() {
    assert!(
        build(97, 824).is_err(),
        "3296 bytes cannot hold 3298 bytes of blocks"
    );
    assert!(
        build(97, 826).is_err(),
        "one whole u32 too many is not rounding"
    );
    assert!(build(97, 100).is_err());
}
