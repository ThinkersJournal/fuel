// SPDX-License-Identifier: MIT OR Apache-2.0
use fuel_ir::error::Result;
use fuel_transformers::generation::LogitsProcessor;

#[test]
fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 0);
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        fuel_transformers::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        fuel_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = [0.1f32, 0.2, 0.3, 0.4];
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_gumbel() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        fuel_transformers::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    let logits = [-1.0f32, 0.0, 0.2, 1.0];
    // Host-side softmax over the f32 slice for the reference distribution.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits.iter().map(|&x| ((x - max) as f64).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let sm: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    let mut counts = vec![0f64; 4];
    let samples = 100000;
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        if (counts[i] - sm[i]).abs() > 0.05 {
            panic!("pr mismatch {counts:?} {sm:?}");
        }
    }
    Ok(())
}

// ---- greedy (`Sampling::ArgMax`) on NaN and ties: the argmax spec rule (ruling 2026-11-25) ----
// A NaN logit means the model produced NaN: sampling must NOT silently pick a token, so it is a
// typed error. With no NaN, ties take the FIRST index (the CPU argmax rule), `-0.0 == +0.0`.

#[test]
fn greedy_sampling_over_a_nan_logit_row_is_a_typed_error() {
    for row in [
        &[1.0f32, f32::NAN, 0.0, 5.0][..],
        &[f32::NAN, 1.0, 5.0],
        &[1.0, 5.0, f32::NAN],
        &[f32::NAN, f32::NAN],
        // a sign-bit NaN sorted BELOW -inf under `total_cmp` and was silently skipped
        &[1.0, -f32::NAN, 5.0],
    ] {
        let mut lp = LogitsProcessor::new(1, None, None);
        let err = lp
            .sample(row)
            .expect_err("a NaN logit row must be an error");
        assert!(format!("{err}").contains("NaN"), "{row:?}: {err}");
    }
}

#[test]
fn greedy_sampling_ties_take_the_first_index() {
    let mut lp = LogitsProcessor::new(1, None, None);
    assert_eq!(lp.sample(&[5.0f32, 5.0, 2.0, 2.0]).unwrap(), 0);
    assert_eq!(lp.sample(&[2.0f32, 5.0, 5.0, 5.0]).unwrap(), 1);
    // -0.0 == +0.0: the first of the tie
    assert_eq!(lp.sample(&[-0.0f32, 0.0, 0.0]).unwrap(), 0);
    assert_eq!(lp.sample(&[0.0f32, -0.0, -0.0]).unwrap(), 0);
    // infinities are ordinary values
    assert_eq!(
        lp.sample(&[f32::NEG_INFINITY, f32::INFINITY, f32::INFINITY])
            .unwrap(),
        1
    );
}

#[test]
fn greedy_sampling_of_an_empty_row_is_still_an_error() {
    let mut lp = LogitsProcessor::new(1, None, None);
    assert!(lp.sample(&[]).is_err());
}
