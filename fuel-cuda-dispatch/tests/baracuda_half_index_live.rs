// SPDX-License-Identifier: MIT OR Apache-2.0
//! Half (f16, bf16) `IndexSelect`, `Gather`, `MaskedFill` and `ScatterAdd` on CUDA through the binding table
//! (GAP-413). The first three are pure 2-byte copies (baracuda's `u16` kernels), so every output element must be
//! BIT-IDENTICAL to the host reference, including `-0.0`, a subnormal, `-inf` and a NaN payload. `ScatterAdd` is
//! baracuda's own half kernel (atomics, undefined order for duplicate indices), checked on small integers that
//! are exact in half, with a duplicate index into one cell.
//!
//! Gate: `require_gpu_run_lock`. Run: `pwsh scripts/gpu-run.ps1 -Project fuel -- cargo test -p fuel-cuda-dispatch
//! --features cuda --test baracuda_half_index_live -- --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_cuda_dispatch::register_baracuda_cuda_kernels;
use fuel_dispatch::{
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
};
use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_memory::{BackendStorage, Storage};
use half::{bf16, f16};

const HALVES: [DType; 2] = [DType::F16, DType::BF16];

/// Edge values: ordinary, -0.0, a subnormal, -inf, a quiet NaN.
const EDGE: [f32; 8] = [
    1.0,
    -0.0,
    1.0e-6,
    f32::NEG_INFINITY,
    f32::NAN,
    2.5,
    -3.0,
    0.0,
];

fn enc(dt: DType, v: &[f32]) -> Vec<u8> {
    match dt {
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

fn storage(dev: &CudaDevice, bytes: &[u8], dt: DType) -> Arc<RwLock<Storage>> {
    let h = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("upload");
    Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(h), dt)))
}

fn out_storage(dev: &CudaDevice, bytes: usize, dt: DType) -> Arc<RwLock<Storage>> {
    let h = CudaStorageBytes::alloc(dev, bytes).expect("alloc");
    Arc::new(RwLock::new(Storage::new(BackendStorage::Cuda(h), dt)))
}

fn read(s: &Arc<RwLock<Storage>>) -> Vec<u8> {
    let g = s.read().unwrap();
    match &g.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("download"),
        _ => panic!("not cuda"),
    }
}

fn table() -> KernelBindingTable {
    let mut t = KernelBindingTable::new();
    register_cuda_kernels(&mut t);
    register_baracuda_cuda_kernels(&mut t);
    t
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Run every alternative bound for `(op, key)`; returns each alternative's result.
fn run_all(
    t: &KernelBindingTable,
    op: OpKind,
    key: &[DType],
    ins: &[Arc<RwLock<Storage>>],
    out: &dyn Fn() -> Arc<RwLock<Storage>>,
    params: &OpParams,
) -> Vec<Result<Vec<u8>, String>> {
    let alts = t.lookup_alternatives(op, key, BackendId::Cuda);
    assert!(
        !alts.is_empty(),
        "{op:?} {key:?}: no CUDA binding (GAP-413 slice not wired)"
    );
    alts.iter()
        .map(|e| {
            let o = out();
            (e.kernel)(ins, &mut [o.clone()], &[], params)
                .map(|_| read(&o))
                .map_err(|e| format!("{e:?}"))
        })
        .collect()
}

#[test]
fn cuda_half_index_select_gather_masked_fill_are_bit_exact() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let t = table();
    let mut bad = Vec::new();
    let mut ran = 0usize;
    for dt in HALVES {
        let src_bits = enc(dt, &EDGE); // 8 elements: viewed as [2, 4]
        let src = storage(&dev, &src_bits, dt);
        let sz = 2usize;
        let col = |i: usize| src_bits[i * sz..(i + 1) * sz].to_vec();

        // IndexSelect along dim 1 of [2, 4]: outer 2, source_dim 4, indices [3, 0, 3], inner 1 -> [2, 3].
        let idx = storage(&dev, &u32_bytes(&[3, 0, 3]), DType::U32);
        let want: Vec<u8> = [3usize, 0, 3, 7, 4, 7]
            .iter()
            .flat_map(|&i| col(i))
            .collect();
        let params = OpParams::IndexSelect {
            outer_count: 2,
            source_dim_size: 4,
            n_indices: 3,
            inner_count: 1,
        };
        for r in run_all(
            &t,
            OpKind::IndexSelect,
            &[dt, DType::U32, dt],
            &[src.clone(), idx],
            &|| out_storage(&dev, 6 * sz, dt),
            &params,
        ) {
            ran += 1;
            match r {
                Ok(got) if got == want => println!("HALFIDX index_select/{dt:?}: OK bit-exact"),
                other => bad.push(format!("index_select/{dt:?}: {other:?}")),
            }
        }

        // Gather along dim 1: source [2, 4], index [2, 3] (per-row), output [2, 3].
        let gi = storage(&dev, &u32_bytes(&[3, 0, 3, 1, 1, 2]), DType::U32);
        let want: Vec<u8> = [3usize, 0, 3, 5, 5, 6]
            .iter()
            .flat_map(|&i| col(i))
            .collect();
        let params = OpParams::Gather {
            source_shape: vec![2, 4],
            output_shape: vec![2, 3],
            dim: 1,
        };
        for r in run_all(
            &t,
            OpKind::Gather,
            &[dt, DType::U32, dt],
            &[src.clone(), gi],
            &|| out_storage(&dev, 6 * sz, dt),
            &params,
        ) {
            ran += 1;
            match r {
                Ok(got) if got == want => println!("HALFIDX gather/{dt:?}: OK bit-exact"),
                other => bad.push(format!("gather/{dt:?}: {other:?}")),
            }
        }

        // MaskedFill with fill = -inf (the attention mask value) and with a NaN fill; mask over 8 elements.
        let mask_v: [u8; 8] = [1, 0, 1, 0, 0, 1, 0, 1];
        let mask = storage(&dev, &mask_v, DType::Bool);
        for (label, fill) in [("-inf", f32::NEG_INFINITY), ("nan", f32::NAN)] {
            let fb = enc(dt, &[fill]);
            let want: Vec<u8> = (0..8)
                .flat_map(|i| if mask_v[i] != 0 { fb.clone() } else { col(i) })
                .collect();
            let params = OpParams::MaskedFill {
                fill_bytes: fb.clone(),
            };
            for r in run_all(
                &t,
                OpKind::MaskedFill,
                &[dt, DType::Bool, dt],
                &[src.clone(), mask.clone()],
                &|| out_storage(&dev, 8 * sz, dt),
                &params,
            ) {
                ran += 1;
                match r {
                    Ok(got) if got == want => {
                        println!("HALFIDX masked_fill({label})/{dt:?}: OK bit-exact")
                    }
                    other => bad.push(format!("masked_fill({label})/{dt:?}: {other:?}")),
                }
            }
        }
    }
    println!("HALFIDX copy ops ran {ran} cases, {} diverge", bad.len());
    assert_eq!(ran, 8, "2 dtypes x 4 copy cases must all run");
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn cuda_half_scatter_add_accumulates_duplicates_and_is_pinned_on_out_of_range() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device (a missing device is a failure)");
    let t = table();
    let mut bad = Vec::new();
    let mut ran = 0usize;
    for dt in HALVES {
        // base [2, 3] = [[1, 2, 3], [4, 5, 6]]; src/indices [2, 2] along dim 1; row 0 sends BOTH to column 1
        // (a duplicate: 2 + 10 + 20 = 32), row 1 sends to columns 0 and 2.
        let base = storage(&dev, &enc(dt, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), dt);
        let upd = storage(&dev, &enc(dt, &[10.0, 20.0, 7.0, 8.0]), dt);
        let idx = storage(&dev, &u32_bytes(&[1, 1, 0, 2]), DType::U32);
        let want = [1.0, 32.0, 3.0, 11.0, 5.0, 14.0];
        let params = OpParams::ScatterAdd {
            base_shape: vec![2, 3],
            src_shape: vec![2, 2],
            dim: 1,
        };
        for r in run_all(
            &t,
            OpKind::ScatterAdd,
            &[dt, DType::U32, dt, dt],
            &[base.clone(), idx, upd.clone()],
            &|| out_storage(&dev, 6 * 2, dt),
            &params,
        ) {
            ran += 1;
            match r.map(|b| dec(dt, &b)) {
                Ok(got) if got == want => println!("HALFIDX scatter_add/{dt:?}: OK {got:?}"),
                other => bad.push(format!("scatter_add/{dt:?}: {other:?}")),
            }
        }
        // Out-of-range index (3 >= base_shape[dim] = 3): MEASURED, printed so the contract states what happens.
        let oob = storage(&dev, &u32_bytes(&[1, 3, 0, 2]), DType::U32);
        for r in run_all(
            &t,
            OpKind::ScatterAdd,
            &[dt, DType::U32, dt, dt],
            &[base.clone(), oob, upd.clone()],
            &|| out_storage(&dev, 6 * 2, dt),
            &params,
        ) {
            ran += 1;
            // MEASURED 2026-11-27 (4070, baracuda alpha.220): the out-of-range update is SKIPPED, the call
            // returns Ok, every other cell accumulates normally (documented baracuda divergence from an error).
            // Conforms to KISS-OPS-6.11-0005/-0016 (scatter MUST skip an out-of-bounds write; the skipped position keeps
            // `dest`). Pins today's CUDA behaviour; the other backends' divergences are GAP-415.
            let want_oob = [1.0, 12.0, 3.0, 11.0, 5.0, 14.0];
            match r.map(|b| dec(dt, &b)) {
                Ok(got) if got == want_oob => {
                    println!("HALFIDX scatter_add_oob/{dt:?}: OK skipped {got:?}")
                }
                other => bad.push(format!("scatter_add_oob/{dt:?}: {other:?}")),
            }
        }
    }
    println!("HALFIDX scatter ran {ran} cases, {} diverge", bad.len());
    assert!(ran >= 4);
    assert!(bad.is_empty(), "{bad:?}");
}

/// `IndexAdd` f16/bf16 (first CUDA keys, GAP-413): `out[idx[i]] += src[i]` on the flat `[outer, axis, inner]`
/// view, with a duplicate index into one row. Small integers are exact in half. The out-of-range case is
/// MEASURED and printed (`HALFIDX index_add_oob/...`): KISS has no `index_add` clause yet (PM ruled 2026-11-27
/// that it SKIPS, pending the clause), so whatever it does today is CURRENT BEHAVIOUR, not the contract.
#[test]
fn cuda_half_index_add_accumulates_duplicates_and_oob_is_measured() {
    fuel_test_support::require_gpu_run_lock();
    let dev = CudaDevice::new(0).expect("cuda device");
    let t = table();
    let mut ran = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for dt in HALVES {
        let base = storage(&dev, &enc(dt, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), dt);
        let src = storage(&dev, &enc(dt, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]), dt);
        let params = OpParams::IndexAdd {
            outer_count: 1,
            base_dim_size: 3,
            n_indices: 3,
            inner_count: 2,
        };
        let idx = storage(&dev, &u32_bytes(&[2, 0, 2]), DType::U32);
        let want = [31.0, 42.0, 3.0, 4.0, 65.0, 86.0];
        for r in run_all(
            &t,
            OpKind::IndexAdd,
            &[dt, DType::U32, dt, dt],
            &[base.clone(), idx, src.clone()],
            &|| out_storage(&dev, 6 * 2, dt),
            &params,
        ) {
            ran += 1;
            match r.map(|b| dec(dt, &b)) {
                Ok(got) if got == want => println!("HALFIDX index_add/{dt:?}: OK {got:?}"),
                other => bad.push(format!("index_add/{dt:?}: {other:?}")),
            }
        }
        // Out-of-range index (3 >= base_dim_size = 3): MEASURED, printed.
        let oob = storage(&dev, &u32_bytes(&[2, 3, 0]), DType::U32);
        for r in run_all(
            &t,
            OpKind::IndexAdd,
            &[dt, DType::U32, dt, dt],
            &[base.clone(), oob, src.clone()],
            &|| out_storage(&dev, 6 * 2, dt),
            &params,
        ) {
            ran += 1;
            println!("HALFIDX index_add_oob/{dt:?}: {:?}", r.map(|b| dec(dt, &b)));
        }
    }
    println!("HALFIDX index_add ran {ran} cases, {} diverge", bad.len());
    assert_eq!(ran, 4, "2 dtypes x (sum + oob) must all run");
    assert!(bad.is_empty(), "{bad:?}");
}
