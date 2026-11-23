// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 169: a quantized weight whose byte count is not a multiple of 4 (an odd `N` of Q8_0, 34-byte
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

/// Row 169 ("kernel read"): the size check is not the only reader of the padded buffer. Realize an odd-N
/// Q8_0 `qmatmul` on the CPU backend (no lowering: the fused node goes to the kernel the executor
/// picks) with the weight stored the way the loader stores it, 3298 block bytes zero-padded to 825 `u32`,
/// and compare with the exact dequantized product.
#[test]
fn odd_n_q8_0_realizes_on_cpu_and_matches_exact_dequant() {
    use fuel_quantized::{BlockQ8_0, GgmlType};
    use fuel_tensor::pipelined_bridge::realize_one_as;
    let (k, n) = (32usize, 97usize);
    let dev = Device::cpu();
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.021).sin() * 0.7)
        .collect();
    let mut blocks = vec![BlockQ8_0::zeros(); n * (k / BlockQ8_0::BLCK_SIZE)];
    BlockQ8_0::from_float(&w_f32, &mut blocks);
    let mut bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * std::mem::size_of::<BlockQ8_0>(),
        )
    }
    .to_vec();
    assert_eq!(bytes.len(), 3298);
    bytes.resize(bytes.len().next_multiple_of(4), 0); // the loader's rounding up: 3300 bytes, 825 u32
    let w_u32: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    assert_eq!(w_u32.len(), 825);
    let mut deq = vec![0f32; n * k];
    BlockQ8_0::to_float(&blocks, &mut deq);

    let a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let x = Tensor::from_f32(a.clone(), Shape::from_dims(&[1, k]), &dev).unwrap();
    let w = x.const_u32_like(w_u32, Shape::from_dims(&[825])).unwrap();
    let y = x.qmatmul(&w, QuantType::Q8_0, k, n).unwrap();
    let t = y.graph_tensor();
    let graph = t.graph().clone();
    let got = realize_one_as::<f32>(&graph, t.id(), &dev)
        .unwrap_or_else(|e| panic!("odd-N Q8_0 realize failed: {e:?}"));
    assert_eq!(got.len(), n);
    for ni in 0..n {
        let want: f32 = (0..k).map(|kk| a[kk] * deq[ni * k + kk]).sum();
        let rel = (got[ni] - want).abs() / got[ni].abs().max(want.abs()).max(f32::MIN_POSITIVE);
        assert!(
            rel < 1e-4,
            "row {ni}: got {}, want {want}, rel {rel}",
            got[ni]
        );
    }
}
