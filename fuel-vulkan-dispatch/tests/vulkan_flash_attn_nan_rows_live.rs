// SPDX-License-Identifier: MIT OR Apache-2.0
// HOSTED-CI NOTE (PIP #715 N2): this test needs a live Vulkan adapter, so the hosted `rust-ci.yml` test steps skip it BY NAME
// (`--skip` list); a rename of any test here MUST be mirrored there, or hosted CI reddens. The GPU runner
// (`device-tests.yml`) runs it whole.
//! FlashAttn on NaN rows through the SHIPPED Vulkan kernels, on every non-CPU adapter, f32 / f16 /
//! bf16, single-pass AND split-KV decode: the masked/NaN-row contract (ruling 2026-11-25): a NaN in
//! Q, or in K/V at a position the query ATTENDS, makes that output NaN (never swallowed); a NaN in a
//! V row whose key is MASKED OUT for the query (causal) is UNSPECIFIED (may be ignored or
//! propagated; the row must not crash and, if it is not NaN, must equal the NaN-free reference);
//! normal rows are unaffected.
//!
//! **NOT MEASURABLE HERE: the FULLY-MASKED-row vector.** The Vulkan flash kernels reject windows and
//! softcap (the wrapper returns an error), and a causal band alone always admits key 0, so a row
//! with no admissible key cannot be built for these kernels.
//!
//! The split-KV path (`decode_splits`: `sq == 1`, `b * hq <= 16`, `2048 <= sk <= 4096`, 8 splits)
//! and its combine pass are exercised by the `split-*` cases, with the NaN in different chunks.
//!
//! The same vectors are `flash_attn_{fused,decomposed}_cpu_follows_the_masked_and_nan_row_contract`
//! in `fuel-tensor/src/lazy.rs`. Gate: `require_gpu_run_lock`; a missing adapter is a FAILURE. Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-vulkan-dispatch --test vulkan_flash_attn_nan_rows_live -- --nocapture --test-threads=1`.

use std::sync::{Arc, RwLock};

use fuel_dispatch::kernel::{KernelBindingTable, OpParams};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use fuel_vulkan_dispatch::vulkan_dispatch::register_vulkan_kernels;
use half::{bf16, f16};

const NAN: f32 = f32::NAN;
const DTYPES: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

#[derive(Debug, Clone, Copy)]
enum Want {
    /// Must be NaN.
    Nan,
    /// Must equal this value within the dtype tolerance.
    Val(f32),
    /// UNSPECIFIED (contract (3), a NaN in a V row masked out for the query): NaN is allowed;
    /// otherwise it must equal this NaN-free reference.
    NanOrVal(f32),
}

struct Case {
    name: String,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    sq: usize,
    sk: usize,
    causal: bool,
    want: Vec<Want>,
}

/// Host reference for one query row (d = 2): `softmax(scale * q.k) . v`, f64.
fn reference_row(q: &[f32], k: &[f32], v: &[f32], sk: usize, scale: f64) -> [f32; 2] {
    let s: Vec<f64> = (0..sk)
        .map(|j| scale * (q[0] as f64 * k[j * 2] as f64 + q[1] as f64 * k[j * 2 + 1] as f64))
        .collect();
    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
    let z: f64 = e.iter().sum();
    let mut out = [0.0f64; 2];
    for j in 0..sk {
        out[0] += e[j] * v[j * 2] as f64;
        out[1] += e[j] * v[j * 2 + 1] as f64;
    }
    [(out[0] / z) as f32, (out[1] / z) as f32]
}

// 4-digit test scale (~1/sqrt(2)), deliberately NOT FRAC_1_SQRT_2: the expected values were computed with exactly 0.7071.
#[allow(clippy::approx_constant)]
const SCALE: f32 = 0.7071;

fn cases() -> Vec<Case> {
    use Want::{Nan as n, NanOrVal as o, Val as s};
    let bq = vec![0.5f32, -0.25, 0.1, 0.9];
    let bk = vec![0.3f32, 0.2, -0.1, 0.6];
    let bv = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut out = vec![
        Case {
            name: "control".into(),
            q: bq.clone(),
            k: bk.clone(),
            v: bv.clone(),
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![s(1.8943309), s(2.894331), s(2.1126559), s(3.1126556)],
        },
        Case {
            name: "nan-q-row1".into(),
            q: vec![0.5, -0.25, NAN, 0.9],
            k: bk.clone(),
            v: bv.clone(),
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![s(1.8943309), s(2.894331), n, n],
        },
        Case {
            name: "nan-k-row1".into(),
            q: bq.clone(),
            k: vec![0.3, 0.2, NAN, 0.6],
            v: bv.clone(),
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![n, n, n, n],
        },
        Case {
            name: "nan-v-row1".into(),
            q: bq.clone(),
            k: bk.clone(),
            v: vec![1.0, 2.0, NAN, 4.0],
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![n, s(2.894331), n, s(3.1126556)],
        },
        Case {
            name: "nan-v-masked-key".into(),
            q: bq.clone(),
            k: bk.clone(),
            v: vec![1.0, 2.0, NAN, 4.0],
            sq: 2,
            sk: 2,
            causal: true,
            want: vec![o(1.0), o(2.0), n, s(3.1126556)],
        },
    ];
    // ---- split-KV decode (sq = 1, sk = 2048, 8 chunks of 256): the NaN sits in different chunks.
    let sk = 2048usize;
    let q = vec![0.5f32, -0.25];
    let k: Vec<f32> = (0..sk * 2)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let v: Vec<f32> = (0..sk * 2)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.02)
        .collect();
    let r = reference_row(&q, &k, &v, sk, SCALE as f64);
    out.push(Case {
        name: "split-control".into(),
        q: q.clone(),
        k: k.clone(),
        v: v.clone(),
        sq: 1,
        sk,
        causal: false,
        want: vec![s(r[0]), s(r[1])],
    });
    out.push(Case {
        name: "split-nan-q".into(),
        q: vec![NAN, -0.25],
        k: k.clone(),
        v: v.clone(),
        sq: 1,
        sk,
        causal: false,
        want: vec![n, n],
    });
    for (tag, at) in [
        ("first-chunk", 3usize),
        ("middle-chunk", 1100),
        ("last-chunk", 2040),
    ] {
        let mut kn = k.clone();
        kn[at * 2] = NAN;
        out.push(Case {
            name: format!("split-nan-k-{tag}"),
            q: q.clone(),
            k: kn,
            v: v.clone(),
            sq: 1,
            sk,
            causal: false,
            want: vec![n, n],
        });
    }
    out
}

fn enc(dt: DType, v: &[f32]) -> Vec<u8> {
    match dt {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => v
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect(),
        DType::BF16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect(),
        _ => unreachable!(),
    }
}

fn dec(dt: DType, b: &[u8]) -> Vec<f32> {
    match dt {
        DType::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        DType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
        _ => unreachable!(),
    }
}

// 4-digit test scale (~1/sqrt(2)), deliberately NOT FRAC_1_SQRT_2: the expected values in `cases` were
// computed with exactly 0.7071 (same posture as the CPU `flash_decompose_vs_reference`).
#[allow(clippy::approx_constant)]
fn run(be: &Arc<VulkanBackend>, dt: DType, c: &Case) -> Result<Vec<f32>, String> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let key = [dt; 4];
    let alts = table.lookup_alternatives(OpKind::FlashAttn, &key, BackendId::Vulkan);
    let alt = alts
        .first()
        .ok_or("no FlashAttn kernel registered for this dtype")?;
    let mut ins = Vec::new();
    for data in [&c.q, &c.k, &c.v] {
        let h = be
            .upload_bytes_handle(&enc(dt, data))
            .map_err(|e| format!("{e:?}"))?;
        ins.push(Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(h),
            dt,
        ))));
    }
    let out = be
        .alloc_bytes_handle(c.sq * 2 * dt.size_in_bytes())
        .map_err(|e| format!("{e:?}"))?;
    let oa = Arc::new(RwLock::new(Storage::new(BackendStorage::Vulkan(out), dt)));
    let layouts = [
        Layout::contiguous(Shape::from_dims(&[1, 1, c.sq, 2])),
        Layout::contiguous(Shape::from_dims(&[1, 1, c.sk, 2])),
        Layout::contiguous(Shape::from_dims(&[1, 1, c.sk, 2])),
        Layout::contiguous(Shape::from_dims(&[1, 1, c.sq, 2])),
    ];
    let params = OpParams::FlashAttn {
        b: 1,
        hq: 1,
        hkv: 1,
        sq: c.sq,
        sk: c.sk,
        d: 2,
        k_len: c.sk,
        softmax_scale: SCALE,
        causal: c.causal,
        window_size_left: None,
        window_size_right: None,
        softcap: None,
    };
    (alt.kernel)(&ins, &mut [Arc::clone(&oa)], &layouts, &params).map_err(|e| format!("{e:?}"))?;
    let bytes = match &oa.read().unwrap().inner {
        BackendStorage::Vulkan(v) => be.download_bytes(v).map_err(|e| format!("{e:?}"))?,
        _ => return Err("not vulkan".into()),
    };
    Ok(dec(dt, &bytes))
}

/// Vectors on which the shipped Vulkan kernels DIVERGE from the contract today, as
/// `"dtype/case"` prefixes. Empty: the poison flag (`flash_attn_*.glsl`, `flash_attn_split_*.glsl`,
/// `flash_attn_combine_*.glsl`) makes every Q/K NaN row NaN. The test reddens if one regresses.
const KNOWN_DIVERGENT: &[&str] = &[];

#[test]
fn vulkan_flash_attn_nan_row_contract() {
    fuel_test_support::require_gpu_run_lock();
    let listed = VulkanBackend::list_devices().expect("list_devices");
    let mut ran = 0;
    for (idx, name, kind) in &listed {
        if kind == "cpu" {
            continue;
        }
        let be = Arc::new(
            VulkanBackend::with_selection(DeviceSelection::Index(*idx))
                .unwrap_or_else(|e| panic!("open adapter {name:?}: {e:?}")),
        );
        let (mut unexpected_div, mut unexpected_agree) = (Vec::new(), Vec::new());
        for dt in DTYPES {
            // f16/bf16 narrow the inputs and the result: a looser tolerance, still far below the
            // distance between a real value and a swallowed zero.
            let tol = match dt {
                DType::F32 => 1e-3,
                DType::F16 => 2e-2,
                _ => 8e-2,
            };
            for c in cases() {
                let got = run(&be, dt, &c)
                    .unwrap_or_else(|e| panic!("{name}/{dt:?}/{}: dispatch failed: {e}", c.name));
                let agree = got.iter().zip(&c.want).all(|(g, w)| match w {
                    Want::Nan => g.is_nan(),
                    Want::Val(x) => !g.is_nan() && (g - x).abs() < tol,
                    Want::NanOrVal(x) => g.is_nan() || (g - x).abs() < tol,
                });
                let key = format!("{dt:?}/{}", c.name);
                println!(
                    "FLASHNAN {name} | {key}: got {got:?} want {:?} {}",
                    c.want,
                    if agree { "OK" } else { "DIVERGES" }
                );
                let known = KNOWN_DIVERGENT.iter().any(|k| key.starts_with(k));
                match (known, agree) {
                    (false, false) => unexpected_div.push(key),
                    (true, true) => unexpected_agree.push(key),
                    _ => {}
                }
            }
        }
        assert!(
            unexpected_div.is_empty() && unexpected_agree.is_empty(),
            "{name}: FlashAttn NaN-row behaviour vs the contract. Unlisted divergences: {unexpected_div:?}. Listed but now agreeing: {unexpected_agree:?}"
        );
        ran += 1;
    }
    assert!(
        ran > 0,
        "no non-CPU Vulkan adapter found (GAP-157: absence is a failure, not a pass)"
    );
}
