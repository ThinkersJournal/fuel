// SPDX-License-Identifier: MIT OR Apache-2.0
//! FlashAttn on masked and NaN rows through EVERY CUDA binding Fuel has at the `[T;4]` key
//! (baracuda `flash_sdpa`, `flash_decoding`, ...), f32/f64/f16/bf16: the contract (ruling 2026-11-25):
//! (1) a FULLY MASKED row (no key admitted by causal/window) is ZEROS; (2) a NaN in Q, or in K/V at a
//! position the query ATTENDS, makes that output NaN (never swallowed); (3) a NaN in a V row whose key is
//! MASKED OUT for the query is UNSPECIFIED (NaN allowed; otherwise it must equal the NaN-free
//! reference); (4) normal rows are unaffected in a mixed batch.
//!
//! **What Fuel binds (measured 2026-11-25, alpha.199):** at the `FlashAttn` `[T;4]` CUDA key there are
//! ZERO bindings for f32/f64 (those resolve to the decomposition / CPU) and exactly ONE for f16 and bf16,
//! `flash_decoding` (a capacity-K DECODE kernel: `seq_q == 1`, no window, no softcap). baracuda's
//! `flash_sdpa` wrappers exist in `fuel-cuda-backend` but are NOT registered in the dispatch table, so
//! the E0k fixes to `flash_sdpa` do not reach Fuel's IR, and the FULLY-MASKED-row case (it needs a window
//! or `sq > sk`) is UNREACHABLE on CUDA: the multi-row cases below report `UNREACHABLE` by design; the
//! `decode-*` cases (`sq == 1`) are the reachable ones.
//!
//! Every registered alternative is run and labelled by its index. A dispatch ERROR (a binding that
//! rejects the params, e.g. a decode kernel for `sq != 1`, or a window) is reported as `UNREACHABLE`,
//! not as a divergence: that is the answer to "does CUDA reach this case from Fuel's IR".
//!
//! The same vectors: CPU `flash_attn_{fused,decomposed}_cpu_follows_the_masked_and_nan_row_contract`
//! (`fuel-tensor/src/lazy.rs`) and the Vulkan `vulkan_flash_attn_nan_rows_live`.
//!
//! Gate: `require_gpu_run_lock`; a missing device is a FAILURE. Run:
//! `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch --features cuda --test baracuda_flash_attn_nan_rows_live -- --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::{
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_ir::{DType, Layout, Shape, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use half::{bf16, f16};

const NAN: f32 = f32::NAN;
const DTYPES: [DType; 4] = [DType::F32, DType::F64, DType::F16, DType::BF16];
const SCALE: f32 = 0.7071;

#[derive(Debug, Clone, Copy)]
enum Want {
    Nan,
    Val(f32),
    /// UNSPECIFIED (contract (3)): NaN allowed, otherwise it must equal this NaN-free reference.
    NanOrVal(f32),
}

struct Case {
    name: &'static str,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    sq: usize,
    sk: usize,
    causal: bool,
    wl: Option<usize>,
    want: Vec<Want>,
}

fn cases() -> Vec<Case> {
    use Want::{Nan as n, NanOrVal as o, Val as s};
    let bq = vec![0.5f32, -0.25, 0.1, 0.9, -0.4, 0.3];
    let bk = vec![0.3f32, 0.2, -0.1, 0.6];
    let bv = vec![1.0f32, 2.0, 3.0, 4.0];
    #[allow(clippy::too_many_arguments)]
    let c = |name, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, sq, sk, causal, wl, want| Case {
        name,
        q,
        k,
        v,
        sq,
        sk,
        causal,
        wl,
        want,
    };
    vec![
        c(
            "control",
            bq[..4].to_vec(),
            bk.clone(),
            bv.clone(),
            2,
            2,
            false,
            None,
            vec![s(1.8943309), s(2.894331), s(2.1126559), s(3.1126556)],
        ),
        // sq=3 > sk=2, causal, window_left=0: row 2 attends nothing (fully masked), rows 0/1 normal.
        c(
            "fully-masked-row-mixed",
            bq.clone(),
            bk.clone(),
            bv.clone(),
            3,
            2,
            true,
            Some(0),
            vec![s(1.0), s(2.0), s(3.0), s(4.0), s(0.0), s(0.0)],
        ),
        c(
            "nan-q-row1",
            vec![0.5, -0.25, NAN, 0.9],
            bk.clone(),
            bv.clone(),
            2,
            2,
            false,
            None,
            vec![s(1.8943309), s(2.894331), n, n],
        ),
        c(
            "nan-k-row1",
            bq[..4].to_vec(),
            vec![0.3, 0.2, NAN, 0.6],
            bv.clone(),
            2,
            2,
            false,
            None,
            vec![n, n, n, n],
        ),
        c(
            "nan-v-row1",
            bq[..4].to_vec(),
            bk.clone(),
            vec![1.0, 2.0, NAN, 4.0],
            2,
            2,
            false,
            None,
            vec![n, s(2.894331), n, s(3.1126556)],
        ),
        c(
            "nan-v-masked-key",
            bq[..4].to_vec(),
            bk.clone(),
            vec![1.0, 2.0, NAN, 4.0],
            2,
            2,
            true,
            None,
            vec![o(1.0), o(2.0), n, s(3.1126556)],
        ),
    ]
    .into_iter()
    .chain(decode_cases())
    .collect()
}

/// Host reference for one decode row (d = 2): `softmax(scale * q.k) . v`, f64.
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

/// The REACHABLE cases: decode rows (`sq == 1`), `sk = 8`, no mask.
fn decode_cases() -> Vec<Case> {
    use Want::{Nan as n, Val as s};
    let sk = 8usize;
    let q = vec![0.5f32, -0.25];
    let k: Vec<f32> = (0..sk * 2)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let v: Vec<f32> = (0..sk * 2)
        .map(|i| ((i * 53 % 97) as f32 - 48.0) * 0.02)
        .collect();
    let r = reference_row(&q, &k, &v, sk, SCALE as f64);
    let mk = |name, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>, want| Case {
        name,
        q,
        k,
        v,
        sq: 1,
        sk,
        causal: false,
        wl: None,
        want,
    };
    let mut kn = k.clone();
    kn[5 * 2] = f32::NAN;
    let mut vn = v.clone();
    vn[3 * 2] = f32::NAN;
    vec![
        mk(
            "decode-control",
            q.clone(),
            k.clone(),
            v.clone(),
            vec![s(r[0]), s(r[1])],
        ),
        mk(
            "decode-nan-q",
            vec![f32::NAN, -0.25],
            k.clone(),
            v.clone(),
            vec![n, n],
        ),
        mk("decode-nan-k", q.clone(), kn, v.clone(), vec![n, n]),
        // NaN in column 0 of a visible V row: column 0 NaN, column 1 finite.
        mk("decode-nan-v", q, k, vn, vec![n, s(r[1])]),
    ]
}

fn enc(dt: DType, v: &[f32]) -> Vec<u8> {
    match dt {
        DType::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F64 => v.iter().flat_map(|x| (*x as f64).to_le_bytes()).collect(),
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
        DType::F64 => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
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
fn run(
    dev: &CudaDevice,
    t: &KernelBindingTable,
    dt: DType,
    alt: usize,
    c: &Case,
) -> Result<Vec<f32>, String> {
    let alts = t.lookup_alternatives(OpKind::FlashAttn, &[dt; 4], BackendId::Cuda);
    let kernel = alts.get(alt).ok_or("no such alternative")?.kernel;
    let mut ins = Vec::new();
    for data in [&c.q, &c.k, &c.v] {
        let h =
            CudaStorageBytes::from_cpu_bytes(dev, &enc(dt, data)).map_err(|e| format!("{e:?}"))?;
        ins.push(Arc::new(RwLock::new(Storage::new(
            BackendStorage::Cuda(h),
            dt,
        ))));
    }
    let out = CudaStorageBytes::alloc(dev, c.sq * 2 * dt.size_in_bytes())
        .map_err(|e| format!("{e:?}"))?;
    let oa = Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(out), dt)));
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
        window_size_left: c.wl,
        window_size_right: None,
        softcap: None,
    };
    kernel(&ins, &mut [Arc::clone(&oa)], &layouts, &params).map_err(|e| format!("{e:?}"))?;
    let g = oa.read().unwrap();
    match &g.inner {
        BackendStorage::Cuda(cb) => Ok(dec(dt, &cb.to_cpu_bytes().map_err(|e| format!("{e:?}"))?)),
        _ => Err("not cuda".into()),
    }
}

/// `"dtype/altN/case"` prefixes on which CUDA DIVERGES today. Filled from the first measured run.
const KNOWN_DIVERGENT: &[&str] = &[];

#[test]
fn cuda_flash_attn_nan_row_contract() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let mut t = KernelBindingTable::new();
    register_cuda_kernels(&mut t);
    register_baracuda_cuda_kernels(&mut t);
    let (mut unexpected_div, mut unexpected_agree) = (Vec::new(), Vec::new());
    let mut ran = 0usize;
    for dt in DTYPES {
        let tol = match dt {
            DType::F16 => 2e-2,
            DType::BF16 => 8e-2,
            _ => 1e-3,
        };
        let n_alts = t
            .lookup_alternatives(OpKind::FlashAttn, &[dt; 4], BackendId::Cuda)
            .len();
        println!("CUDAFLASH {dt:?}: {n_alts} alternative(s)");
        for alt in 0..n_alts {
            for c in cases() {
                let key = format!("{dt:?}/alt{alt}/{}", c.name);
                let got = match run(&dev, &t, dt, alt, &c) {
                    Ok(g) => g,
                    Err(e) => {
                        println!(
                            "CUDAFLASH {key}: UNREACHABLE ({})",
                            e.lines().next().unwrap_or("")
                        );
                        continue;
                    }
                };
                ran += 1;
                let agree = got.iter().zip(&c.want).all(|(g, w)| match w {
                    Want::Nan => g.is_nan(),
                    Want::Val(x) => !g.is_nan() && (g - x).abs() < tol,
                    Want::NanOrVal(x) => g.is_nan() || (g - x).abs() < tol,
                });
                println!(
                    "CUDAFLASH {key}: got {got:?} want {:?} {}",
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
    }
    println!(
        "CUDAFLASH ran {ran} reachable cases, {} diverge",
        unexpected_div.len()
    );
    assert!(ran > 0, "no CUDA FlashAttn binding ran");
    assert!(
        unexpected_div.is_empty() && unexpected_agree.is_empty(),
        "CUDA FlashAttn NaN-row behaviour vs the contract. Unlisted divergences: {unexpected_div:?}. Listed but now agreeing: {unexpected_agree:?}"
    );
}
