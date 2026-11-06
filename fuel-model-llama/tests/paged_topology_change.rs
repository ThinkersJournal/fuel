// SPDX-License-Identifier: MIT OR Apache-2.0
//! A `TopologyChanged` on the held paged-decode plan must not advance the pool twice.
//!
//! `LlamaModel::forward_paged_step_persistent_inv_freq` advances the session by one slot
//! (`advance_paged_session`) BEFORE it realizes. If the realize then fails with
//! `TopologyChanged` (any `register_runtime_kernel` / `extend_global_bindings` landing between
//! plan and realize bumps the process-wide generation), the old fallback re-ran the whole step
//! through `forward_paged_step_inv_freq`, which advances again: one token, two KV slots.
//!
//! This file is its OWN test binary on purpose: `bump_topology_generation()` mutates a
//! process-wide counter, and every other test in a shared binary would observe it.

use std::sync::Arc;

use fuel_ir::DType;
use fuel_kv_pool::KvGeometry;
use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};
use fuel_tensor::Device;
use fuel_tensor::inference_context::PagedDecodePlan;
use fuel_tensor::kv_block_pool_device::DeviceKvPool;
use fuel_tensor::lazy::LayerWeights;

/// Deterministic tiny weights sized to `cfg` (mirrors lazy.rs's test builder).
fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
    let mut s: u32 = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    };
    let mut vec_of = |n: usize| -> Arc<[f32]> {
        let v: Vec<f32> = (0..n).map(|_| next()).collect();
        Arc::from(v)
    };
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    LlamaWeights {
        instance: fuel_tensor::decode_shape::ModelInstanceId::next(),
        token_embedding: vec_of(cfg.vocab_size * cfg.dim),
        layers: (0..cfg.n_layers)
            .map(|_| LayerWeights {
                attn_q: vec_of(cfg.dim * cfg.dim).into(),
                attn_q_bias: None,
                attn_k: vec_of(cfg.dim * kv_dim).into(),
                attn_k_bias: None,
                attn_v: vec_of(cfg.dim * kv_dim).into(),
                attn_v_bias: None,
                attn_o: vec_of(cfg.dim * cfg.dim).into(),
                ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            })
            .collect(),
        final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
        output: vec_of(cfg.dim * cfg.vocab_size).into(),
    }
}

fn tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 16,
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        head_dim: 4,
        ffn_dim: 16,
        norm_eps: 1e-5,
        rope_base: 10000.0,
    }
}

fn fresh_pool(cfg: &LlamaConfig, dev: &Device) -> DeviceKvPool {
    let geom = KvGeometry {
        n_layers: cfg.n_layers,
        num_blocks: 32,
        block_size: 4,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        elem_size: 4,
    };
    DeviceKvPool::new(geom, DType::F32, dev).unwrap()
}

/// Reference: the re-planning step, no persistent session, no generation bump.
fn reference_logits(
    model: &LlamaModel,
    cfg: &LlamaConfig,
    dev: &Device,
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    let mut pool = fresh_pool(cfg, dev);
    let session = pool.core_mut().open();
    tokens
        .iter()
        .map(|&t| model.forward_paged_step(t, &mut pool, session).unwrap())
        .collect()
}

#[test]
fn a_topology_change_advances_the_pool_once_and_keeps_the_logits() {
    let cfg = tiny_cfg();
    let model = LlamaModel {
        config: cfg.clone(),
        weights: tiny_weights(&cfg, 4242),
    };
    let dev = Device::cpu();
    let tokens = [1u32, 2, 3, 4];
    let cap = 8usize;

    let want = reference_logits(&model, &cfg, &dev, &tokens);

    // Persistent: three tokens build and rebind the plan, then the generation is bumped so
    // the fourth token's rebind fails with `TopologyChanged`.
    let mut pool = fresh_pool(&cfg, &dev);
    let session = pool.core_mut().open();
    let mut held = None;
    let mut got = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        if i == 3 {
            assert!(
                held.is_some(),
                "a plan must be held before the bump (else nothing can go stale)"
            );
            fuel_dispatch::dispatch::bump_topology_generation();
        }
        got.push(
            model
                .forward_paged_step_persistent_inv_freq(
                    t,
                    &mut pool,
                    session,
                    cap,
                    PagedDecodePlan::PlanOnce,
                    &mut held,
                    None,
                )
                .unwrap(),
        );
    }

    assert_eq!(
        pool.core().filled_tokens(session),
        Some(tokens.len()),
        "four tokens fed must leave four slots filled (a TopologyChanged fallback advanced twice)"
    );
    assert_eq!(
        got[3].len(),
        want[3].len(),
        "the recovered step must return a full logits row"
    );
    // Tolerance: measured correct-path drift is exactly 0 (the same kernels run in the same
    // order), so 1e-5 is a rounding allowance, far below any wrong-position divergence.
    for (j, (g, w)) in got[3].iter().zip(want[3].iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "logit {j} after the topology change: {g} vs {w}"
        );
    }
}
