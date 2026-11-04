// SPDX-License-Identifier: MIT OR Apache-2.0
//! `f64 -> f16` and `f64 -> bf16` must round to nearest, ties to even, exactly.
//!
//! Reported 2026-11-03 against `half` 2.7.1: `half::f16::from_f64(0.04518128032270489)` gave
//! `0.045166015625`, but the value lies 5.9e-9 ABOVE the midpoint of its two f16 neighbours
//! (`0.045166015625` and `0.0451965332...`), so the nearest f16 is the upper one. Fuel converts through
//! `half::f16::from_f64` at many sites (see the PR), so the conversion is checked here against two
//! independent exact oracles that do not call `half` for the rounding decision.
//!
//! * Oracle A: integer arithmetic. Split the f64 into significand and exponent, shift the significand to
//!   the target quantum, and round half to even from the discarded bits.
//! * Oracle B: nearest-neighbour search. Binary-search the monotone f16/bf16 bit patterns for the largest
//!   value not above `|x|`, then compare the two exact distances (the subtractions are exact: neighbours
//!   are within a factor of two) and break ties to the even pattern.
//!   They are asserted equal on every probed input before either is used to judge `half`.

use half::{bf16, f16};

/// A binary16 / bfloat16-shaped target: `m` stored mantissa bits, `bias`, `max_e` the largest finite
/// exponent, `inf` the bit pattern of +infinity.
#[derive(Clone, Copy)]
struct Fmt {
    m: u32,
    bias: i32,
    max_e: i32,
    inf: u32,
    total: u32,
}

const F16: Fmt = Fmt {
    m: 10,
    bias: 15,
    max_e: 15,
    inf: 0x7C00,
    total: 16,
};
const BF16: Fmt = Fmt {
    m: 7,
    bias: 127,
    max_e: 127,
    inf: 0x7F80,
    total: 16,
};

/// Oracle A. Returns the target bit pattern for a finite or infinite `x`; NaN is handled by the callers.
fn exact_shift(x: f64, f: Fmt) -> u32 {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u32) << (f.total - 1);
    let e = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if e == 0x7ff {
        assert_eq!(frac, 0, "NaN is not an input of the exact oracle");
        return sign | f.inf;
    }
    let (sig, exp2) = if e == 0 {
        (frac as u128, 1 - 1023 - 52)
    } else {
        ((frac | (1u64 << 52)) as u128, e - 1023 - 52)
    };
    if sig == 0 {
        return sign;
    }
    let e_top = (127 - sig.leading_zeros() as i32) + exp2; // floor(log2 |x|)
    let emin = 1 - f.bias;
    let q_exp = e_top.max(emin) - f.m as i32; // exponent of one unit in the last place
    let s = q_exp - exp2; // right shift that turns sig into a count of quanta
    assert!(s > 0, "target finer than f64: not a supported shape");
    let n: u128 = if s >= 127 {
        0
    } else {
        let q = sig >> s;
        let rem = sig & ((1u128 << s) - 1);
        let half = 1u128 << (s - 1);
        if rem > half || (rem == half && (q & 1) == 1) {
            q + 1
        } else {
            q
        }
    };
    if n == 0 {
        return sign;
    }
    let mag: u128 = if e_top < emin {
        n // subnormal: the count of quanta IS the encoding; n == 2^m is the smallest normal
    } else {
        let mut ee = e_top;
        let mut nn = n;
        if nn == 1u128 << (f.m + 1) {
            ee += 1;
            nn = 1u128 << f.m;
        }
        if ee > f.max_e {
            return sign | f.inf;
        }
        (((ee + f.bias) as u128) << f.m) | (nn - (1u128 << f.m))
    };
    sign | (mag as u32).min(f.inf)
}

/// The exact value of a non-negative target bit pattern; `inf` is mapped to the first unrepresentable
/// power of two so it can serve as the upper neighbour of the largest finite value.
fn value_of(b: u32, f: Fmt) -> f64 {
    if b >= f.inf {
        return 2f64.powi(f.max_e + 1);
    }
    let e = (b >> f.m) as i32;
    let m = (b & ((1 << f.m) - 1)) as f64;
    if e == 0 {
        m * 2f64.powi(1 - f.bias - f.m as i32)
    } else {
        (1.0 + m / (1u32 << f.m) as f64) * 2f64.powi(e - f.bias)
    }
}

/// Oracle B.
fn exact_search(x: f64, f: Fmt) -> u32 {
    let sign = if x.is_sign_negative() {
        1u32 << (f.total - 1)
    } else {
        0
    };
    let a = x.abs();
    if a.is_infinite() {
        return sign | f.inf;
    }
    // Largest bit pattern in [0, inf) with value <= a.
    let (mut lo, mut hi) = (0u32, f.inf); // value_of(lo) <= a < value_of(hi) holds for a < 2^(max_e+1)
    if a >= value_of(f.inf, f) {
        return sign | f.inf;
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if value_of(mid, f) <= a {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (vl, vh) = (value_of(lo, f), value_of(lo + 1, f));
    let (dl, dh) = (a - vl, vh - a); // exact (Sterbenz), neighbours are within a factor of two
    let pick = if dl < dh {
        lo
    } else if dh < dl {
        lo + 1
    } else if lo % 2 == 0 {
        lo
    } else {
        lo + 1
    };
    sign | pick.min(f.inf)
}

fn next_up(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}
fn next_down(x: f64) -> f64 {
    f64::from_bits(x.to_bits() - 1)
}

/// Deterministic xorshift, so a failure reproduces.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Inputs around every tie between two adjacent target values: the exact midpoint, f64-adjacent values,
/// and offsets from 2^-40 up to a quarter of the spacing (the reported value sits at about 2^-12.3 of it),
/// on both sides, for positive and negative signs; plus seeded random values across the whole range.
fn probe_inputs(f: Fmt) -> Vec<f64> {
    let mut v = Vec::new();
    let fracs: [f64; 12] = [
        0.0,
        2f64.powi(-40),
        2f64.powi(-30),
        2f64.powi(-20),
        2f64.powi(-16),
        2f64.powi(-14),
        2f64.powi(-12),
        2f64.powi(-10),
        2f64.powi(-8),
        2f64.powi(-6),
        2f64.powi(-4),
        2f64.powi(-2),
    ];
    let top = f.inf; // pair (b, b+1) for b in 0..inf, the last pair's upper neighbour is the overflow bound
    // The f16 set is small enough for every pair; bf16 has 32k patterns too (16 bits), so also every pair.
    for b in 0..top {
        let (lo, hi) = (value_of(b, f), value_of(b + 1, f));
        let mid = (lo + hi) / 2.0;
        let width = hi - lo;
        v.push(mid);
        v.push(next_up(mid));
        v.push(next_down(mid));
        for &fr in &fracs[1..] {
            v.push(mid + fr * width);
            v.push(mid - fr * width);
        }
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let max = value_of(f.inf, f) * 1.01;
    for _ in 0..400_000 {
        v.push(rng.unit() * max); // uniform
        v.push(max * (rng.unit() * -40.0).exp2()); // log-uniform down to 2^-40 of the range (subnormals)
    }
    // The value from the report, and a few neighbours.
    v.extend([
        0.04518128032270489,
        1.0,
        65504.0,
        65519.99999,
        65520.0,
        6.0e-8,
        5.96e-8,
        3.0e-8,
    ]);
    let neg: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v
}

#[test]
fn the_two_exact_oracles_agree_with_each_other_on_every_probe() {
    for (name, f) in [("f16", F16), ("bf16", BF16)] {
        let inputs = probe_inputs(f);
        assert!(
            inputs.len() > 1_000_000,
            "{name}: probe set is too small to mean anything"
        );
        let mut disagree = Vec::new();
        for &x in &inputs {
            let (a, b) = (exact_shift(x, f), exact_search(x, f));
            if a != b {
                disagree.push((x, a, b));
            }
        }
        assert!(
            disagree.is_empty(),
            "{name}: oracles A and B disagree on {} of {} inputs, first {:?}: one of them is wrong, so \
             neither can judge `half`",
            disagree.len(),
            inputs.len(),
            &disagree[..disagree.len().min(3)]
        );
    }
}

#[test]
fn the_reported_value_rounds_up_in_both_oracles() {
    let x = 0.04518128032270489_f64;
    // The upper neighbour is 0.0451965332... = 0x29 << ... ; derive it from the oracle and check it by value.
    let up = exact_shift(x, F16);
    let down = up - 1;
    assert_eq!(up, exact_search(x, F16));
    assert!(value_of(up, F16) > x && x > value_of(down, F16));
    assert!(
        (value_of(up, F16) - x) < (x - value_of(down, F16)),
        "the upper neighbour must be the nearer one"
    );
    assert_eq!(value_of(down, F16), 0.045166015625);
}

/// How far `x` sits from the tie between its two target neighbours, as a fraction of the spacing there
/// (0 = exactly on the tie, 0.5 = on a representable value). `None` for zero or out-of-range inputs.
fn tie_fraction(x: f64, f: Fmt) -> Option<f64> {
    let a = x.abs();
    if a == 0.0 || a >= value_of(f.inf, f) {
        return None;
    }
    let (mut lo, mut hi) = (0u32, f.inf);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if value_of(mid, f) <= a {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (vl, vh) = (value_of(lo, f), value_of(lo + 1, f));
    Some(((a - (vl + vh) / 2.0).abs()) / (vh - vl))
}

/// Counts mismatches and describes them by how far the input sat from its tie.
fn judge(name: &str, f: Fmt, convert: impl Fn(f64) -> u32) -> usize {
    let inputs = probe_inputs(f);
    {
        // Rate on ordinary data only: the 800_000 seeded random positives (they sit just before the 8
        // hand-picked values and the negated copy at the end of the vector).
        let n = inputs.len() / 2;
        let rand_part = &inputs[n - 8 - 800_000..n - 8];
        let wrong = rand_part
            .iter()
            .filter(|&&x| convert(x) != exact_shift(x, f))
            .count();
        eprintln!(
            "{name}: RANDOM values only: {wrong} of {} round wrongly ({:.4}%)",
            rand_part.len(),
            100.0 * wrong as f64 / rand_part.len() as f64
        );
    }
    let mut bad: Vec<(f64, u32, u32)> = Vec::new();
    for &x in &inputs {
        let want = exact_shift(x, f);
        let got = convert(x);
        if got != want {
            bad.push((x, got, want));
        }
    }
    if !bad.is_empty() {
        // Where do the failures sit? Histogram by floor(log2(distance to the tie / spacing)): a band that
        // stops at some 2^-k says how many low bits of the input the conversion ignores.
        let mut hist = std::collections::BTreeMap::<i32, usize>::new();
        let mut farthest = 0.0f64;
        for (x, _, _) in &bad {
            if let Some(fr) = tie_fraction(*x, f) {
                farthest = farthest.max(fr);
                let k = if fr == 0.0 {
                    -99
                } else {
                    fr.log2().floor() as i32
                };
                *hist.entry(k).or_default() += 1;
            }
        }
        eprintln!(
            "{name}: failures by log2(distance to tie / spacing) (-99 = exactly on the tie): {hist:?}; farthest from a tie: {farthest:e} of the spacing"
        );
        // The same inputs split by kind: the random tail (the last 800_000 of the positives) is what ordinary
        // data looks like; the rest is dense around ties on purpose.
        eprintln!(
            "{name}: {} of {} inputs round wrongly; first 5 (x, got bits, exact bits): {:?}",
            bad.len(),
            inputs.len(),
            &bad[..bad.len().min(5)]
        );
    }
    bad.len()
}

/// DEFECT DOCUMENTATION. `half` 2.7.1 mis-rounds near-ties (see `fuel_ir::round`). This test asserts the defect
/// is STILL there; when upstream is fixed it goes red: delete `fuel-ir/src/round.rs`, point the sites back at
/// `half`, and delete this pair.
#[test]
#[cfg_attr(
    target_arch = "aarch64",
    ignore = "half 2.7.1 uses the hardware fcvt on aarch64, which rounds f64->f16 exactly: the defect is x86-only (macOS CI)"
)]
fn half_f16_defect_is_still_present() {
    let n = judge("half::f16::from_f64", F16, |x| {
        f16::from_f64(x).to_bits() as u32
    });
    assert!(
        n > 0,
        "half::f16::from_f64 now rounds every probed input correctly: remove fuel_ir::round (f16)"
    );
}

#[test]
fn half_bf16_defect_is_still_present() {
    let n = judge("half::bf16::from_f64", BF16, |x| {
        bf16::from_f64(x).to_bits() as u32
    });
    assert!(
        n > 0,
        "half::bf16::from_f64 now rounds every probed input correctly: remove fuel_ir::round (bf16)"
    );
}

#[test]
fn fuel_round_f64_to_half_is_exact_against_both_oracles() {
    let n16 = judge("fuel_ir::round::f16_from_f64", F16, |x| {
        fuel_ir::round::f16_from_f64(x).to_bits() as u32
    });
    let nb = judge("fuel_ir::round::bf16_from_f64", BF16, |x| {
        fuel_ir::round::bf16_from_f64(x).to_bits() as u32
    });
    assert_eq!(
        (n16, nb),
        (0, 0),
        "fuel_ir::round mis-rounds: f16 {n16}, bf16 {nb}"
    );
    // Oracle B (nearest-neighbour search) independently, on the same inputs.
    for (f, conv) in [
        (
            F16,
            (|x| fuel_ir::round::f16_from_f64(x).to_bits() as u32) as fn(f64) -> u32,
        ),
        (BF16, |x| fuel_ir::round::bf16_from_f64(x).to_bits() as u32),
    ] {
        for &x in &probe_inputs(f) {
            assert_eq!(conv(x), exact_search(x, f), "oracle B disagrees at {x:e}");
        }
    }
}

#[test]
fn fuel_conversion_paths_round_f16_and_bf16_correctly() {
    use fuel_ir::{DType, WithDType, scalar::Scalar};
    let n16 = judge("Fuel WithDType f16", F16, |x| {
        <f16 as WithDType>::from_f64(x).to_bits() as u32
    });
    let nb = judge("Fuel WithDType bf16", BF16, |x| {
        <bf16 as WithDType>::from_f64(x).to_bits() as u32
    });
    let ns = judge(
        "Fuel Scalar::from_f64 F16",
        F16,
        |x| match Scalar::from_f64(x, DType::F16) {
            Ok(Scalar::F16(h)) => h.to_bits() as u32,
            other => panic!("Scalar::from_f64(F16) gave {other:?}"),
        },
    );
    let nsb = judge(
        "Fuel Scalar::from_f64 BF16",
        BF16,
        |x| match Scalar::from_f64(x, DType::BF16) {
            Ok(Scalar::BF16(h)) => h.to_bits() as u32,
            other => panic!("Scalar::from_f64(BF16) gave {other:?}"),
        },
    );
    assert_eq!(
        (n16, nb, ns, nsb),
        (0, 0, 0, 0),
        "Fuel's own f64 -> f16/bf16 entry points mis-round: WithDType f16 {n16}, bf16 {nb}; Scalar f16 {ns}, bf16 {nsb}"
    );
}

/// NaN and infinities are not rounding questions but must survive the same entry points.
#[test]
fn nan_and_infinity_survive_the_conversion() {
    use fuel_ir::round::{bf16_from_f64, f16_from_f64};
    assert!(f16_from_f64(f64::NAN).is_nan());
    assert!(bf16_from_f64(f64::NAN).is_nan());
    assert_eq!(f16_from_f64(f64::INFINITY).to_bits(), 0x7C00);
    assert_eq!(f16_from_f64(f64::NEG_INFINITY).to_bits(), 0xFC00);
    assert_eq!(bf16_from_f64(f64::INFINITY).to_bits(), 0x7F80);
}
