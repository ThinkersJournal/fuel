// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row 172 item 3 (the conv epilogue): the epilogue is an ELEMENTWISE RECIPE over
//! the conv output, not a closed list of activations (pm-kernels, 2026-11-25;
//! CireSnave board 290). This file is the expressibility evidence for the union
//! bar: cuDNN's conv-bias-activation set (relu, tanh, sigmoid, elu, clipped-relu,
//! swish) plus the usual kernel-language ones (leaky-relu, relu6, softplus,
//! hardswish, hardsigmoid, mish, selu, softsign, hardtanh, gelu).
//!
//! Each activation is composed ONLY from ops Fuel has today and realized on the
//! CPU, then compared with an f64 closed form. A form that cannot be composed
//! would be a MISSING OP, named as a row; this run found none (the test
//! `every_named_activation_is_expressible_and_matches_its_closed_form` is the
//! census: it counts the set and asserts all matched).
//!
//! Characterisation, not born red: the ops already existed. The sabotage sibling
//! (`a_wrong_recipe_is_caught_at_the_stated_tolerance`) proves the tolerance can
//! fail: a leaky-relu with the wrong slope must exceed it.

use fuel_ir::Shape;
use fuel_tensor::lazy::Tensor;

/// A spread of inputs: negative, zero, small, large, both signs.
fn inputs() -> Vec<f32> {
    let mut v: Vec<f32> = (-24..=24).map(|i| i as f32 * 0.25).collect(); // -6 .. 6
    v.extend([0.01, -0.01, 0.5, -0.5, 1e-3, -1e-3]);
    v
}

fn tensor(x: &[f32]) -> Tensor {
    Tensor::from_f32(
        x.to_vec(),
        Shape::from_dims(&[x.len()]),
        &fuel_tensor::Device::cpu(),
    )
    .unwrap()
}

fn sigmoid_r(x: &Tensor) -> Tensor {
    // 1 / (1 + exp(-x)), from Exp / Neg / Recip / AddScalar only.
    x.neg().exp().add_scalar(1.0).recip()
}

fn softplus_r(x: &Tensor) -> Tensor {
    // The stable form: max(x, 0) + log(1 + exp(-|x|)).
    let zero = x.mul_scalar(0.0);
    x.maximum(&zero)
        .unwrap()
        .add(&x.abs().neg().exp().add_scalar(1.0).log())
        .unwrap()
}

fn leaky_r(x: &Tensor, slope: f64) -> Tensor {
    // max(x, slope * x), valid for 0 <= slope <= 1.
    x.maximum(&x.mul_scalar(slope)).unwrap()
}

fn elu_r(x: &Tensor, alpha: f64) -> Tensor {
    let zero = x.mul_scalar(0.0);
    let neg_branch = x.exp().add_scalar(-1.0).mul_scalar(alpha);
    x.gt(&zero).unwrap().where_cond(x, &neg_branch).unwrap()
}

fn selu_r(x: &Tensor) -> Tensor {
    const ALPHA: f64 = 1.673_263_242_354_377_3;
    const SCALE: f64 = 1.050_700_987_355_480_5;
    elu_r(x, ALPHA).mul_scalar(SCALE)
}

fn softsign_r(x: &Tensor) -> Tensor {
    x.div(&x.abs().add_scalar(1.0)).unwrap()
}

fn hardsigmoid_r(x: &Tensor) -> Tensor {
    x.add_scalar(3.0).clamp(0.0, 6.0).mul_scalar(1.0 / 6.0)
}

fn hardswish_r(x: &Tensor) -> Tensor {
    x.mul(&hardsigmoid_r(x)).unwrap()
}

fn mish_r(x: &Tensor) -> Tensor {
    x.mul(&softplus_r(x).tanh()).unwrap()
}

fn sigmoid_f(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus_f(x: f64) -> f64 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}
/// `erf` by composite Simpson's rule over `[0, x]` (2000 panels). Accuracy bound: the
/// Simpson error is `(x * h^4 / 180) * max|f4|` with `h = x / 2000`, `f(t) = exp(-t^2)` and
/// `max|f4| = 12` (the fourth derivative at `t = 0`); for the largest argument used here
/// (`|x| <= 6 / sqrt(2)` = 4.25) that is about `4e-12`, six orders below the `2e-5`
/// comparison tolerance, so the tolerance is set by the f32 recipe, not by this reference
/// (checked against known values by `the_erf_reference_is_accurate_far_below_the_tolerance`).
fn erf_f(x: f64) -> f64 {
    erf_n(x, 2000)
}

fn erf_n(x: f64, n: usize) -> f64 {
    let h = x / n as f64;
    let f = |t: f64| (-t * t).exp();
    let mut s = f(0.0) + f(x);
    for i in 1..n {
        s += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    2.0 / std::f64::consts::PI.sqrt() * s * h / 3.0
}

struct Case {
    name: &'static str,
    recipe: Box<dyn Fn(&Tensor) -> Tensor>,
    reference: Box<dyn Fn(f64) -> f64>,
}

fn cases() -> Vec<Case> {
    let c =
        |name, recipe: Box<dyn Fn(&Tensor) -> Tensor>, reference: Box<dyn Fn(f64) -> f64>| Case {
            name,
            recipe,
            reference,
        };
    vec![
        c("relu", Box::new(|x| x.relu()), Box::new(|v| v.max(0.0))),
        c("tanh", Box::new(|x| x.tanh()), Box::new(|v| v.tanh())),
        c("sigmoid", Box::new(sigmoid_r), Box::new(sigmoid_f)),
        c(
            "elu(1.0)",
            Box::new(|x| elu_r(x, 1.0)),
            Box::new(|v| if v > 0.0 { v } else { v.exp() - 1.0 }),
        ),
        c(
            "clipped_relu(0,6) = relu6",
            Box::new(|x| x.clamp(0.0, 6.0)),
            Box::new(|v| v.clamp(0.0, 6.0)),
        ),
        c(
            "swish = x*sigmoid(x) from primitives",
            Box::new(|x| x.mul(&sigmoid_r(x)).unwrap()),
            Box::new(|v| v * sigmoid_f(v)),
        ),
        c(
            "silu (the op)",
            Box::new(|x| x.silu()),
            Box::new(|v| v * sigmoid_f(v)),
        ),
        c(
            "leaky_relu(0.01)",
            Box::new(|x| leaky_r(x, 0.01)),
            Box::new(|v| if v > 0.0 { v } else { 0.01 * v }),
        ),
        c(
            "leaky_relu(0.2)",
            Box::new(|x| leaky_r(x, 0.2)),
            Box::new(|v| if v > 0.0 { v } else { 0.2 * v }),
        ),
        c("softplus", Box::new(softplus_r), Box::new(softplus_f)),
        c(
            "hardsigmoid",
            Box::new(hardsigmoid_r),
            Box::new(|v| ((v + 3.0).clamp(0.0, 6.0)) / 6.0),
        ),
        c(
            "hardswish",
            Box::new(hardswish_r),
            Box::new(|v| v * ((v + 3.0).clamp(0.0, 6.0)) / 6.0),
        ),
        c(
            "mish",
            Box::new(mish_r),
            Box::new(|v| v * softplus_f(v).tanh()),
        ),
        c(
            "selu",
            Box::new(selu_r),
            Box::new(|v| {
                let (a, s) = (1.673_263_242_354_377_3_f64, 1.050_700_987_355_480_5_f64);
                s * if v > 0.0 { v } else { a * (v.exp() - 1.0) }
            }),
        ),
        c(
            "softsign",
            Box::new(softsign_r),
            Box::new(|v| v / (1.0 + v.abs())),
        ),
        c(
            "hardtanh(-1,1)",
            Box::new(|x| x.clamp(-1.0, 1.0)),
            Box::new(|v| v.clamp(-1.0, 1.0)),
        ),
        c(
            "gelu (erf, the op)",
            Box::new(|x| x.gelu()),
            Box::new(|v| 0.5 * v * (1.0 + erf_f(v / std::f64::consts::SQRT_2))),
        ),
    ]
}

/// Absolute-or-relative closeness at the f32 working precision (the recipes run in
/// f32 and compound a few roundings; calibrated by the sabotage sibling below).
fn close(got: f32, want: f64) -> bool {
    let (g, w) = (got as f64, want);
    (g - w).abs() <= 2e-5 + 2e-5 * w.abs()
}

#[test]
fn every_named_activation_is_expressible_and_matches_its_closed_form() {
    let xs = inputs();
    let mut bad: Vec<String> = Vec::new();
    let all = cases();
    for case in &all {
        let got = (case.recipe)(&tensor(&xs)).realize_f32();
        assert_eq!(got.len(), xs.len(), "{}: length", case.name);
        for (i, (&g, &x)) in got.iter().zip(xs.iter()).enumerate() {
            let want = (case.reference)(x as f64);
            if !close(g, want) {
                bad.push(format!("{} at x={x}: got {g}, want {want}", case.name));
                if bad.len() > 12 {
                    break;
                }
            }
            let _ = i;
        }
    }
    // Census: the set is the union bar named in the module doc (17 forms), and every
    // one is expressible from today's ops.
    assert_eq!(
        all.len(),
        17,
        "the activation set changed: update the census"
    );
    assert!(bad.is_empty(), "mismatches: {bad:#?}");
}

/// Sabotage sibling: the tolerance can fail. A leaky-relu built with the WRONG slope
/// (0.2 instead of 0.01) must be flagged against the 0.01 closed form.
#[test]
fn a_wrong_recipe_is_caught_at_the_stated_tolerance() {
    let xs = inputs();
    let got = leaky_r(&tensor(&xs), 0.2).realize_f32();
    let flagged = got
        .iter()
        .zip(xs.iter())
        .filter(|(g, x)| {
            let (g, x) = (**g, **x);
            let want = if x > 0.0 { x as f64 } else { 0.01 * x as f64 };
            !close(g, want)
        })
        .count();
    assert!(
        flagged > 10,
        "the wrong slope must be visible; flagged {flagged}"
    );
}

/// The gelu reference's own error is far below the tolerance: `erf_f` against values
/// known to 16 digits (`erf(0.5)`, `erf(1)`, `erf(3)`) and its oddness; at the largest
/// argument (4.25) it is checked by SELF-CONVERGENCE (2000 panels vs 16000 agree to 1e-10).
#[test]
fn the_erf_reference_is_accurate_far_below_the_tolerance() {
    for (x, want) in [
        (0.5_f64, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (3.0, 0.999_977_909_503_001_4),
    ] {
        assert!((erf_f(x) - want).abs() < 1e-10, "erf({x}) = {}", erf_f(x));
        assert!((erf_f(-x) + want).abs() < 1e-10, "erf(-{x}) odd");
    }
    assert!(
        (erf_f(4.25) - erf_n(4.25, 16_000)).abs() < 1e-10,
        "Simpson has converged at 4.25"
    );
}
