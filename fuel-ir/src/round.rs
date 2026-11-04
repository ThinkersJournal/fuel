// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact `f64 -> f16` and `f64 -> bf16` conversion, round to nearest, ties to even.
//!
//! **Why this exists.** `half::f16::from_f64` and `half::bf16::from_f64` (half 2.7.1, the version this
//! repository locks) mis-round values that lie very close to, but not exactly on, the tie between two
//! neighbouring target values: they behave as if the value were exactly on the tie and round to even. Measured
//! against two independent exact oracles (`fuel-ir/tests/f64_to_half_rounding.rs`): 0.0056% of ordinary random
//! f64 inputs for f16 and 0.0060% for bf16, up to about 6.1e-5 (f16) and 1.2e-4 (bf16) of the spacing away from
//! a tie, and never exactly on a tie. Every f64 -> f16 / bf16 conversion in Fuel goes through here instead.
//!
//! The method is integer arithmetic only: split the f64 into significand and exponent, shift the significand
//! down to the target's quantum, and round half to even from the discarded bits (all of them, including the
//! sticky ones). NaN keeps its sign and top payload bits and is forced quiet; infinities map to infinities;
//! values beyond the largest finite target value round to infinity exactly where round-to-nearest says so;
//! subnormal results are rounded at their own, coarser quantum.
//!
//! When upstream `half` is fixed, `half_*_defect_is_still_present` in the test file above goes red and tells
//! whoever sees it to delete this module and call `half` again.

use half::{bf16, f16};

/// Round an f64 to the nearest f16 (ties to even), exactly.
#[inline]
pub fn f16_from_f64(x: f64) -> f16 {
    f16::from_bits(round_to_bits(x, 10, 15, 15, 0x7C00))
}

/// Round an f64 to the nearest bf16 (ties to even), exactly.
#[inline]
pub fn bf16_from_f64(x: f64) -> bf16 {
    bf16::from_bits(round_to_bits(x, 7, 127, 127, 0x7F80))
}

/// The shared core. `m` is the number of stored mantissa bits, `bias` the exponent bias, `max_e` the largest
/// finite exponent and `inf` the bit pattern of +infinity (exponent all ones, mantissa zero).
#[inline]
fn round_to_bits(x: f64, m: u32, bias: i32, max_e: i32, inf: u16) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let e = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if e == 0x7ff {
        if frac == 0 {
            return sign | inf;
        }
        // NaN: keep the top payload bits that fit, and set the quiet bit so the result is never an infinity.
        return sign | inf | (1u16 << (m - 1)) | (frac >> (52 - m)) as u16;
    }
    // x = sig * 2^exp2 with sig an integer below 2^53.
    let (sig, exp2) = if e == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), e - 1075)
    };
    if sig == 0 {
        return sign;
    }
    let e_top = (63 - sig.leading_zeros() as i32) + exp2; // floor(log2 |x|)
    let emin = 1 - bias;
    // One unit in the last place of the target at this magnitude (coarser than normal below `emin`).
    let q_exp = e_top.max(emin) - m as i32;
    let s = q_exp - exp2; // always > 0: the target is much coarser than an f64
    debug_assert!(s > 0);
    let n = rounded_quanta(sig, s as u32);
    if n == 0 {
        return sign;
    }
    sign | encode_magnitude(n, e_top, emin, m, bias, max_e, inf)
}

/// `sig / 2^s` rounded to nearest, ties to even, as a count of quanta. For s >= 64 the whole significand is
/// below half a quantum, so the count is 0 (and `1 << s` would not fit a u64).
#[inline]
fn rounded_quanta(sig: u64, s: u32) -> u64 {
    if s >= 64 {
        return 0;
    }
    let q = sig >> s;
    let rem = sig & ((1u64 << s) - 1);
    let half = 1u64 << (s - 1);
    if rem > half || (rem == half && (q & 1) == 1) {
        q + 1
    } else {
        q
    }
}

/// The magnitude bits (no sign) for `n` quanta at binade `e_top`; overflow saturates to `inf`.
#[inline]
fn encode_magnitude(n: u64, e_top: i32, emin: i32, m: u32, bias: i32, max_e: i32, inf: u16) -> u16 {
    let mag: u64 = if e_top < emin {
        // Subnormal range: the count of quanta IS the encoding, and n == 2^m is exactly the smallest normal.
        n
    } else {
        let (mut ee, mut nn) = (e_top, n);
        if nn == 1u64 << (m + 1) {
            // Rounding carried into the next binade.
            ee += 1;
            nn = 1u64 << m;
        }
        if ee > max_e {
            return inf;
        }
        (((ee + bias) as u64) << m) | (nn - (1u64 << m))
    };
    (mag as u16).min(inf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cases_that_motivated_this_module() {
        // On a tie: ties to even, both directions.
        assert_eq!(f16_from_f64(1.0 + 2f64.powi(-11)).to_bits(), 0x3C00); // between 0x3C00 (even) and 0x3C01
        assert_eq!(f16_from_f64(1.0 + 3.0 * 2f64.powi(-11)).to_bits(), 0x3C02); // between 0x3C01 and 0x3C02 (even)
        // A hair above a tie must go UP, not to the even neighbour.
        assert_eq!(
            f16_from_f64(1.0 + 2f64.powi(-11) + 2f64.powi(-40)).to_bits(),
            0x3C01
        );
        // The smallest f16 subnormal's half: below it rounds to zero, a hair above it rounds up.
        assert_eq!(f16_from_f64(2f64.powi(-25)).to_bits(), 0x0000);
        assert_eq!(
            f16_from_f64(2f64.powi(-25) + 2f64.powi(-70)).to_bits(),
            0x0001
        );
        // Overflow boundary: 65520 is the first value that rounds to infinity.
        assert_eq!(f16_from_f64(65519.999).to_bits(), 0x7BFF);
        assert_eq!(f16_from_f64(65520.0).to_bits(), 0x7C00);
        assert_eq!(f16_from_f64(-65520.0).to_bits(), 0xFC00);
    }

    #[test]
    fn nan_inf_and_signed_zero() {
        assert!(f16_from_f64(f64::NAN).is_nan());
        assert!(bf16_from_f64(f64::NAN).is_nan());
        assert_eq!(f16_from_f64(f64::INFINITY).to_bits(), 0x7C00);
        assert_eq!(f16_from_f64(f64::NEG_INFINITY).to_bits(), 0xFC00);
        assert_eq!(f16_from_f64(-0.0).to_bits(), 0x8000);
        assert_eq!(bf16_from_f64(-0.0).to_bits(), 0x8000);
        assert_eq!(bf16_from_f64(f64::INFINITY).to_bits(), 0x7F80);
    }
}
