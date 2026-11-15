// SPDX-License-Identifier: MIT OR Apache-2.0
//! The CPU `QMatMul` float route (board 307, PM ruling 2026-11-15): weights stay quantized in
//! memory, a tile of rows is dequantized to f32 and multiplied against the **f32 activations**;
//! no quantized-by-quantized product, no integer dot.
//!
//! Each format's kernel is measured against an f64 product of the same f32 activations and the
//! weights dequantized by `GgmlType::to_float`. The sabotage control is the other route: the same
//! product with the activations first rounded to 8 bits per 32-block (what ggml's integer dot
//! consumes). It lies well outside the float tolerance, so a kernel that quantized its activations
//! would fail here instead of passing for the wrong reason.

use fuel_cpu_backend::CpuStorageBytes;
use fuel_cpu_backend::byte_kernels as bk;
use fuel_quantized::iq_grid_types::BlockIq3S;
use fuel_quantized::iq4::{BlockIq4Nl, BlockIq4Xs};
use fuel_quantized::{
    BlockQ3K, BlockQ4_0, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, blocks_as_bytes,
};

type Kernel = fn(
    &CpuStorageBytes,
    &CpuStorageBytes,
    &mut CpuStorageBytes,
    usize,
    usize,
    usize,
    usize,
) -> fuel_ir::Result<()>;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| self.next() as f32 / (1u32 << 31) as f32 * 2.0 - 1.0)
            .collect()
    }
}

/// Blocks of `T` from float weights (formats with a quantizer).
fn quantized<T: GgmlType>(rng: &mut Lcg, n: usize, k: usize) -> Vec<u8> {
    let w = rng.unit(n * k);
    let mut blocks = vec![T::zeros(); n * k / T::BLCK_SIZE];
    T::from_float(&w, &mut blocks);
    blocks_as_bytes(&blocks).to_vec()
}

/// Blocks of `T` from random bytes with the leading f16 `d` set to a small finite value (the IQ
/// formats keep `d` first). Every other field decodes for any byte value.
fn random_blocks<T: GgmlType>(rng: &mut Lcg, n: usize, k: usize) -> Vec<u8> {
    let count = n * k / T::BLCK_SIZE;
    let size = std::mem::size_of::<T>();
    let mut bytes: Vec<u8> = (0..count * size).map(|_| rng.next() as u8).collect();
    let d = half::f16::from_f32(0.02).to_le_bytes();
    for b in 0..count {
        bytes[b * size..b * size + 2].copy_from_slice(&d);
    }
    bytes
}

fn dequant<T: GgmlType>(bytes: &[u8]) -> Vec<f32> {
    let mut blocks = vec![T::zeros(); bytes.len() / std::mem::size_of::<T>()];
    bytemuck::cast_slice_mut::<T, u8>(&mut blocks).copy_from_slice(bytes);
    let mut out = vec![0.0_f32; blocks.len() * T::BLCK_SIZE];
    T::to_float(&blocks, &mut out);
    out
}

/// Round each 32-block of `x` to 8 bits (absmax scale), as an integer-dot path would.
fn activations_as_8_bit(x: &[f32]) -> Vec<f32> {
    x.chunks(32)
        .flat_map(|c| {
            let amax = c.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            c.iter()
                .map(move |v| if d == 0.0 { 0.0 } else { (v / d).round() * d })
        })
        .collect()
}

fn product(x: &[f32], w: &[f32], (batch, m, n, k): (usize, usize, usize, usize)) -> Vec<f64> {
    let mut out = vec![0.0_f64; batch * m * n];
    for b in 0..batch {
        for r in 0..m {
            for o in 0..n {
                out[(b * m + r) * n + o] = (0..k)
                    .map(|c| f64::from(x[(b * m + r) * k + c]) * f64::from(w[o * k + c]))
                    .sum();
            }
        }
    }
    out
}

/// Largest `|got - want|` scaled by the sum of absolute terms (the rounding-error scale).
fn worst_scaled(
    got: &[f32],
    want: &[f64],
    x: &[f32],
    w: &[f32],
    (batch, m, n, k): (usize, usize, usize, usize),
) -> f64 {
    let mut worst = 0.0_f64;
    for b in 0..batch {
        for r in 0..m {
            for o in 0..n {
                let scale: f64 = (0..k)
                    .map(|c| (f64::from(x[(b * m + r) * k + c]) * f64::from(w[o * k + c])).abs())
                    .sum();
                let i = (b * m + r) * n + o;
                worst = worst.max((f64::from(got[i]) - want[i]).abs() / scale.max(1e-9));
            }
        }
    }
    worst
}

fn check<T: GgmlType>(
    name: &str,
    kernel: Kernel,
    weights: impl Fn(&mut Lcg, usize, usize) -> Vec<u8>,
) {
    // 19 rows: not a multiple of the 8-row tile, so the ragged tile is exercised.
    for &(batch, m, n, k) in &[(1, 1, 19, 512), (2, 5, 19, 512), (1, 3, 8, 256)] {
        let mut rng = Lcg(0x5EED ^ (n * k) as u64);
        let wbytes = weights(&mut rng, n, k);
        let w = dequant::<T>(&wbytes);
        let x = rng.unit(batch * m * k);
        let mut out = CpuStorageBytes::from_zero_bytes(batch * m * n * 4);
        kernel(
            &CpuStorageBytes::from_slice(&x),
            &CpuStorageBytes::from_bytes(&wbytes),
            &mut out,
            batch,
            m,
            n,
            k,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let got: &[f32] = out.as_slice().unwrap();
        let dims = (batch, m, n, k);
        let float_err = worst_scaled(got, &product(&x, &w, dims), &x, &w, dims);
        assert!(
            float_err < 2e-6,
            "{name} {dims:?}: float route error {float_err:e}"
        );
        // Control: the 8-bit-activation product is far from the kernel's output.
        let xq = activations_as_8_bit(&x);
        let int_err = worst_scaled(got, &product(&xq, &w, dims), &x, &w, dims);
        assert!(
            int_err > 100.0 * float_err.max(1e-9) && int_err > 1e-4,
            "{name} {dims:?}: the kernel is indistinguishable from an 8-bit-activation product ({int_err:e} vs {float_err:e})"
        );
    }
}

#[test]
fn q4_0() {
    check::<BlockQ4_0>("q4_0", bk::qmatmul_q4_0_f32, |r, n, k| {
        quantized::<BlockQ4_0>(r, n, k)
    });
}
#[test]
fn q8_0() {
    check::<BlockQ8_0>("q8_0", bk::qmatmul_q8_0_f32, |r, n, k| {
        quantized::<BlockQ8_0>(r, n, k)
    });
}
#[test]
fn q4_k() {
    check::<BlockQ4K>("q4_k", bk::qmatmul_q4_k_m_f32, |r, n, k| {
        quantized::<BlockQ4K>(r, n, k)
    });
}
#[test]
fn q6_k() {
    check::<BlockQ6K>("q6_k", bk::qmatmul_q6k_f32, |r, n, k| {
        quantized::<BlockQ6K>(r, n, k)
    });
}
#[test]
fn q3_k() {
    check::<BlockQ3K>("q3_k", bk::qmatmul_q3k_f32, |r, n, k| {
        quantized::<BlockQ3K>(r, n, k)
    });
}
#[test]
fn q5_k() {
    check::<BlockQ5K>("q5_k", bk::qmatmul_q5k_f32, |r, n, k| {
        quantized::<BlockQ5K>(r, n, k)
    });
}
#[test]
fn iq4_nl() {
    check::<BlockIq4Nl>("iq4_nl", bk::qmatmul_iq4_nl_f32, |r, n, k| {
        random_blocks::<BlockIq4Nl>(r, n, k)
    });
}
#[test]
fn iq4_xs() {
    check::<BlockIq4Xs>("iq4_xs", bk::qmatmul_iq4_xs_f32, |r, n, k| {
        random_blocks::<BlockIq4Xs>(r, n, k)
    });
}
#[test]
fn iq3_s() {
    check::<BlockIq3S>("iq3_s", bk::qmatmul_iq3_s_f32, |r, n, k| {
        random_blocks::<BlockIq3S>(r, n, k)
    });
}

/// A `k` that is not a whole number of blocks is a typed error, never a short read.
#[test]
fn a_partial_block_is_an_error() {
    let w = CpuStorageBytes::from_zero_bytes(4 * std::mem::size_of::<BlockIq4Xs>());
    let x = CpuStorageBytes::from_zero_bytes(100 * 4);
    let mut out = CpuStorageBytes::from_zero_bytes(4);
    assert!(bk::qmatmul_iq4_xs_f32(&x, &w, &mut out, 1, 1, 1, 100).is_err());
}
