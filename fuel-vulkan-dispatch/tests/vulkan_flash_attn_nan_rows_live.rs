// SPDX-License-Identifier: MIT OR Apache-2.0
//! FlashAttn on NaN rows through the SHIPPED Vulkan kernel, on every non-CPU adapter (f32): the
//! masked/NaN-row contract (ruling 2026-11-25): a NaN in Q, or in K/V at a position the query
//! ATTENDS, makes that output NaN (never swallowed); a NaN in a V row whose key is MASKED OUT for
//! the query (causal) is ignored by the fused kernels; normal rows are unaffected.
//!
//! **NOT MEASURABLE HERE: the FULLY-MASKED-row vector.** The Vulkan flash kernel rejects windows and
//! softcap (`wl.is_some() || wr.is_some() || softcap.is_some()` is an error), and a causal band
//! alone always admits key 0, so a row with no admissible key cannot be built for this kernel.
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

const NAN: f32 = f32::NAN;

struct Case {
    name: &'static str,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    sq: usize,
    sk: usize,
    causal: bool,
    /// `None` = NaN, `Some(x)` = that value within 1e-3.
    want: Vec<Option<f32>>,
}

fn cases() -> Vec<Case> {
    let bq = vec![0.5f32, -0.25, 0.1, 0.9];
    let bk = vec![0.3f32, 0.2, -0.1, 0.6];
    let bv = vec![1.0f32, 2.0, 3.0, 4.0];
    let s = Some;
    vec![
        Case {
            name: "control",
            q: bq.clone(),
            k: bk.clone(),
            v: bv.clone(),
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![s(1.8943309), s(2.894331), s(2.1126559), s(3.1126556)],
        },
        Case {
            name: "nan-q-row1",
            q: vec![0.5, -0.25, NAN, 0.9],
            k: bk.clone(),
            v: bv.clone(),
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![s(1.8943309), s(2.894331), None, None],
        },
        Case {
            name: "nan-k-row1",
            q: bq.clone(),
            k: vec![0.3, 0.2, NAN, 0.6],
            v: bv.clone(),
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![None, None, None, None],
        },
        Case {
            name: "nan-v-row1",
            q: bq.clone(),
            k: bk.clone(),
            v: vec![1.0, 2.0, NAN, 4.0],
            sq: 2,
            sk: 2,
            causal: false,
            want: vec![None, s(2.894331), None, s(3.1126556)],
        },
        Case {
            name: "nan-v-masked-key",
            q: bq.clone(),
            k: bk.clone(),
            v: vec![1.0, 2.0, NAN, 4.0],
            sq: 2,
            sk: 2,
            causal: true,
            want: vec![s(1.0), s(2.0), None, s(3.1126556)],
        },
    ]
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run(be: &Arc<VulkanBackend>, c: &Case) -> Result<Vec<f32>, String> {
    let mut table = KernelBindingTable::new();
    register_vulkan_kernels(&mut table);
    let key = [DType::F32; 4];
    let alts = table.lookup_alternatives(OpKind::FlashAttn, &key, BackendId::Vulkan);
    let alt = alts.first().ok_or("no FlashAttn f32 kernel registered")?;
    let mut ins = Vec::new();
    for data in [&c.q, &c.k, &c.v] {
        let h = be
            .upload_bytes_handle(&f32_bytes(data))
            .map_err(|e| format!("{e:?}"))?;
        ins.push(Arc::new(RwLock::new(Storage::new(
            BackendStorage::Vulkan(h),
            DType::F32,
        ))));
    }
    let out = be
        .alloc_bytes_handle(c.sq * 2 * 4)
        .map_err(|e| format!("{e:?}"))?;
    let oa = Arc::new(RwLock::new(Storage::new(
        BackendStorage::Vulkan(out),
        DType::F32,
    )));
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
        softmax_scale: 0.7071,
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
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// Vectors on which the shipped Vulkan kernel DIVERGES from the contract today (name prefixes),
/// measured 2026-11-25: `nan-q-row1`: a NaN Q row returns ZEROS (swallowed: the row max stays -inf, the
/// row is skipped, as the old CPU kernel did); `nan-v-masked-key`: a NaN in a V row whose key is masked
/// out for the query still PROPAGATES (row 0 is NaN; the contract says the fused kernels ignore it).
/// The test reddens when either is fixed: delete the entry then.
const KNOWN_DIVERGENT: &[&str] = &["nan-q-row1", "nan-v-masked-key"];

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
        for c in cases() {
            let got =
                run(&be, &c).unwrap_or_else(|e| panic!("{name}/{}: dispatch failed: {e}", c.name));
            let agree = got.iter().zip(&c.want).all(|(g, w)| match w {
                None => g.is_nan(),
                Some(x) => !g.is_nan() && (g - x).abs() < 1e-3,
            });
            println!(
                "FLASHNAN {name} | {}: got {got:?} want {:?} {}",
                c.name,
                c.want,
                if agree { "OK" } else { "DIVERGES" }
            );
            let known = KNOWN_DIVERGENT.iter().any(|k| c.name.starts_with(k));
            match (known, agree) {
                (false, false) => unexpected_div.push(c.name),
                (true, true) => unexpected_agree.push(c.name),
                _ => {}
            }
        }
        assert!(
            unexpected_div.is_empty() && unexpected_agree.is_empty(),
            "{name}: FlashAttn NaN-row behaviour vs the contract. Unlisted divergences: {unexpected_div:?}. Listed but now agreeing: {unexpected_agree:?}"
        );
        ran += 1;
    }
    assert!(ran > 0, "no non-CPU Vulkan adapter found (GAP-157)");
}
