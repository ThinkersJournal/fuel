// SPDX-License-Identifier: MIT OR Apache-2.0
//! `sample_logits` / `spec_argmax` follow the `sample_argmax` contract (cluster ruling 2026-11-26,
//! after Lightbulb found a NaN row returning a token): a NaN anywhere is a typed `Err`, ties take the
//! FIRST index, infinities are ordinary values for greedy, and the temperature path refuses any
//! non-finite logit as `sample_with_params` does. The old `max_by(partial_cmp -> Equal)` picked an
//! arbitrary index on NaN and the LAST index on a tie.

use fuel_tensor::lazy::{SamplingStrategy, sample_logits, spec_argmax};

fn greedy(row: &[f32]) -> fuel_ir::Result<u32> {
    sample_logits(row, SamplingStrategy::Greedy, &mut 1)
}

fn temp(row: &[f32]) -> fuel_ir::Result<u32> {
    sample_logits(
        row,
        SamplingStrategy::Temperature { temp: 1.0, seed: 1 },
        &mut 1,
    )
}

#[test]
fn a_nan_row_is_an_err_wherever_the_nan_sits() {
    for row in [
        vec![f32::NAN, 1.0, 2.0],
        vec![1.0, f32::NAN, 2.0],
        vec![1.0, 5.0, f32::NAN],
        vec![f32::NAN],
        vec![f32::NAN, f32::NAN],
        // A sign-bit NaN is still NaN.
        vec![1.0, f32::from_bits(0xFFC0_0000), 0.5],
    ] {
        assert!(greedy(&row).is_err(), "greedy {row:?}");
        assert!(temp(&row).is_err(), "temperature {row:?}");
        assert!(spec_argmax(&row).is_err(), "spec_argmax {row:?}");
    }
}

#[test]
fn ties_take_the_first_index_and_signed_zero_is_equal() {
    assert_eq!(greedy(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
    assert_eq!(greedy(&[7.0, 7.0, 7.0]).unwrap(), 0);
    assert_eq!(greedy(&[-0.0, 0.0, -0.0]).unwrap(), 0);
    assert_eq!(spec_argmax(&[2.0, 9.0, 9.0]).unwrap(), 1);
    assert_eq!(
        sample_logits(
            &[0.0, 4.0, 4.0],
            SamplingStrategy::Temperature { temp: 0.0, seed: 3 },
            &mut 3
        )
        .unwrap(),
        1
    );
}

#[test]
fn infinities_are_ordinary_for_greedy_and_refused_by_temperature() {
    assert_eq!(greedy(&[1.0, f32::INFINITY, 2.0]).unwrap(), 1);
    assert_eq!(greedy(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).unwrap(), 0);
    assert_eq!(greedy(&[f32::NEG_INFINITY, -3.0]).unwrap(), 1);
    assert!(temp(&[1.0, f32::INFINITY]).is_err());
    assert!(temp(&[f32::NEG_INFINITY, 1.0]).is_err());
}

#[test]
fn an_empty_row_is_an_err_not_a_panic() {
    assert!(greedy(&[]).is_err());
    assert!(temp(&[]).is_err());
    assert!(spec_argmax(&[]).is_err());
}

/// Positive control: a finite row still samples (the Err arms are not unconditional).
#[test]
fn a_finite_row_still_samples() {
    assert_eq!(greedy(&[0.1, 0.2, 0.0, 0.3, 0.15]).unwrap(), 3);
    assert!(temp(&[0.1, 0.2, 0.0, 0.3, 0.15]).unwrap() < 5);
}

/// Retained sabotage sibling: the REMOVED implementation, copied here verbatim, picks a token from a
/// NaN row and the LAST of tied maxima, so the assertions above can fail.
#[test]
fn the_removed_max_by_picks_garbage_so_the_tests_above_can_fail() {
    fn legacy(l: &[f32]) -> u32 {
        l.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0 as u32
    }
    assert_eq!(legacy(&[1.0, 3.0, 3.0, 2.0]), 2, "last tied index");
    // A NaN row yields SOME index with no error.
    let _ = legacy(&[1.0, f32::NAN, 2.0]);
}
