// SPDX-License-Identifier: MIT OR Apache-2.0
//! The opt-out of the unwired-quant decline (`FUEL_GGML_KEEP_QUANTIZED=1`, hotfix #706).
//!
//! In a `vulkan` build `WeightStorage::from_ggml_bytes` declines Q3_K, Q5_K, IQ4_NL, IQ4_XS and IQ3_S
//! (the Vulkan `QMatMul` has no arm for them) unless the environment variable is `1`; the variable is read
//! ONCE per process. An environment read is process state, so each case runs in a child process: this
//! one test re-executes itself with `GGML_KEEP_TEST_CHILD` naming the case.
//!
//! Cases: `default` (variable removed: the five are declined, the three wired types are kept), `keep`
//! (variable `1`: all eight kept as blocks), `once-decline` (starts unset, then the variable is set to `1`
//! inside the process: still declined) and `once-keep` (starts `1`, then set to `0`: still kept).
//! Sabotage kept as the test's meaning: ignoring the variable fails `keep` and `once-keep`; flipping the
//! default fails `default` and `once-decline`; reading it on every call fails both `once-*` cases.
#![cfg(feature = "vulkan")]

use fuel_ir::quantized::GgmlDType;
use fuel_tensor::lazy::{KEEP_QUANTIZED_ENV, WeightStorage, declines_unwired_vulkan_quant};

/// `(type, bytes per block, elements per block)`.
const FIVE: [(GgmlDType, usize, usize); 5] = [
    (GgmlDType::Q3K, 110, 256),
    (GgmlDType::Q5K, 176, 256),
    (GgmlDType::Iq4Nl, 18, 32),
    (GgmlDType::Iq4Xs, 136, 256),
    (GgmlDType::Iq3S, 110, 256),
];
/// Wired on Vulkan: kept in every case.
const WIRED: [(GgmlDType, usize, usize); 3] = [
    (GgmlDType::Q8_0, 34, 32),
    (GgmlDType::Q4K, 144, 256),
    (GgmlDType::Q6K, 210, 256),
];

fn kept((dt, bytes_per_block, block): (GgmlDType, usize, usize)) -> bool {
    let bytes = vec![0_u8; bytes_per_block * (512 / block) * 3];
    match WeightStorage::from_ggml_bytes(dt, &bytes, 512, 3) {
        Some(w) => {
            assert_eq!(w.resident_bytes(), bytes.len(), "{dt:?} kept as blocks");
            true
        }
        None => false,
    }
}

fn expect_five(keep: bool, when: &str) {
    for t in WIRED {
        assert!(kept(t), "{when}: wired {:?} must always be kept", t.0);
    }
    for t in FIVE {
        assert_eq!(kept(t), keep, "{when}: {:?} kept = {keep} expected", t.0);
    }
    assert_eq!(declines_unwired_vulkan_quant(), !keep, "{when}: the query");
}

fn run_child(case: &str) {
    match case {
        "default" => expect_five(false, "default"),
        "keep" => expect_five(true, "keep"),
        "once-decline" => {
            expect_five(false, "before the variable is set");
            // SAFETY: this child runs one test on one thread; nothing else reads the environment.
            unsafe { std::env::set_var(KEEP_QUANTIZED_ENV, "1") };
            expect_five(false, "after the variable is set to 1 (read once)");
        }
        "once-keep" => {
            expect_five(true, "before the variable is changed");
            // SAFETY: as above.
            unsafe { std::env::set_var(KEEP_QUANTIZED_ENV, "0") };
            expect_five(true, "after the variable is set to 0 (read once)");
        }
        other => panic!("unknown case {other}"),
    }
}

fn spawn(case: &str, env: Option<&str>) {
    let mut cmd = std::process::Command::new(std::env::current_exe().expect("this test exe"));
    cmd.args([
        "--exact",
        "the_opt_out_is_read_once_and_the_default_declines",
        "--nocapture",
        "--test-threads=1",
    ])
    .env("GGML_KEEP_TEST_CHILD", case)
    .env_remove(KEEP_QUANTIZED_ENV);
    if let Some(v) = env {
        cmd.env(KEEP_QUANTIZED_ENV, v);
    }
    let out = cmd.output().expect("spawn the child");
    assert!(
        out.status.success(),
        "case `{case}` failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // Control: the child really ran the test (a zero-match filter exits 0).
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("1 passed"),
        "case `{case}`: the child ran no test:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn the_opt_out_is_read_once_and_the_default_declines() {
    if let Ok(case) = std::env::var("GGML_KEEP_TEST_CHILD") {
        run_child(&case);
        return;
    }
    spawn("default", None);
    spawn("keep", Some("1"));
    spawn("once-decline", None);
    spawn("once-keep", Some("1"));
}
