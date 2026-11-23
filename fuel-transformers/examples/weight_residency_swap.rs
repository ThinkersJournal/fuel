// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 172: two DIFFERENT models back and forth on one Vulkan device, to see what the device's
//! weight residency does to VRAM and to the logits when the model in use changes.
//!
//! Model A is the Qwen3-0.6B Q4_K_M GGUF of `qwen3_vulkan_profile`; model B is TinyLlama-1.1B-Chat Q4_0
//! (`TINYLLAMA_GGUF` + `TINYLLAMA_DIR`, default under `C:\models`). Each "visit" makes a fresh
//! `InferenceContext` and `KvCache` (what a per-sequence consumer does), prefills a prompt and decodes
//! a few tokens, printing an FNV hash of every call's logits, so two runs (residency off, residency
//! on) can be compared bit for bit. `FUEL_WEIGHT_RESIDENCY=0` is the off setting.
//!
//! `cargo build --release -p fuel-transformers --features vulkan,dev-tools --example weight_residency_swap`,
//! then run it through `gpu-run`. `SWAP_ROUNDS` (default 3) is the number of A,B round trips.

use fuel_ir::DType;
use fuel_tensor::inference_context::{DecodeSession, InferenceContext, KvCache};
use fuel_transformers::dev_tools::qwen3_yardstick::{PROMPTS, load_model};
use fuel_transformers::models::lazy_llama_full::LlamaFullConfig;
use fuel_transformers::models::lazy_quantized_llama::QuantizedLlama3Model;
use std::time::Instant;

fn fnv(logits: &[f32]) -> u64 {
    logits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
    })
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in v.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0 as u32
}

fn vram_used_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

type Forward<'a> = dyn Fn(
        &[u32],
        &mut KvCache,
        &mut InferenceContext,
        &mut Option<DecodeSession>,
    ) -> Result<Vec<f32>, String>
    + 'a;

/// One visit: fresh context and cache, prefill, then `decodes` decode steps. Returns the logits hashes
/// (prefill first) and the prefill and first-decode wall times in ms.
fn visit(
    dev: &fuel_tensor::Device,
    (layers, kv_heads, head_dim): (usize, usize, usize),
    prompt: &[u32],
    decodes: usize,
    forward: &Forward<'_>,
) -> (Vec<u64>, f64, f64) {
    let mut cache = KvCache::with_capacity(layers, kv_heads, head_dim, 256, DType::F32, dev)
        .expect("KvCache::with_capacity");
    let mut ctx = InferenceContext::new(dev.clone());
    let mut session: Option<DecodeSession> = None;
    let mut hashes = Vec::new();
    let t = Instant::now();
    let logits = forward(prompt, &mut cache, &mut ctx, &mut session).expect("prefill");
    let prefill_ms = t.elapsed().as_secs_f64() * 1e3;
    hashes.push(fnv(&logits));
    let mut next = argmax(&logits);
    let mut first_decode_ms = 0.0;
    for i in 0..decodes {
        let t = Instant::now();
        let logits = forward(&[next], &mut cache, &mut ctx, &mut session).expect("decode");
        if i == 0 {
            first_decode_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        hashes.push(fnv(&logits));
        next = argmax(&logits);
    }
    (hashes, prefill_ms, first_decode_ms)
}

fn main() {
    let rounds: usize = std::env::var("SWAP_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let (qwen, tokenizer, qcfg) = load_model().expect("load Qwen3 GGUF");
    let tl_dir = std::env::var("TINYLLAMA_DIR")
        .unwrap_or_else(|_| "C:/models/TinyLlama-1.1B-Chat-v1.0".to_string());
    let tl_gguf = std::env::var("TINYLLAMA_GGUF").unwrap_or_else(|_| {
        "C:/models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_0.gguf".to_string()
    });
    let tcfg = LlamaFullConfig::from_hf_json_str(
        &std::fs::read_to_string(format!("{tl_dir}/config.json")).expect("TinyLlama config.json"),
    )
    .expect("parse TinyLlama config");
    let tiny = QuantizedLlama3Model::from_gguf(&tl_gguf, &tcfg).expect("load TinyLlama GGUF");
    let dev = fuel_tensor::vulkan_backend::new_device_from_env().expect("vulkan device");

    let formatted = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        PROMPTS[0]
    );
    let qprompt: Vec<u32> = tokenizer
        .encode(formatted, true)
        .expect("encode")
        .get_ids()
        .to_vec();
    let tprompt: Vec<u32> = (0..qprompt.len() as u32)
        .map(|i| 100 + (i * 37) % 30_000)
        .collect();
    let qshape = (
        qcfg.num_hidden_layers,
        qcfg.num_key_value_heads,
        qcfg.head_dim,
    );
    let tshape = (
        tcfg.num_hidden_layers,
        tcfg.num_key_value_heads,
        tcfg.head_dim,
    );

    let base = vram_used_mib().unwrap_or(0);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(base));
    let (stop2, peak2) = (stop.clone(), peak.clone());
    let sampler = std::thread::spawn(move || {
        while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(v) = vram_used_mib() {
                peak2.fetch_max(v, std::sync::atomic::Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    });

    println!(
        "residency env FUEL_WEIGHT_RESIDENCY={:?}; qwen prompt {} tokens, tinyllama prompt {} tokens; {rounds} round trips",
        std::env::var("FUEL_WEIGHT_RESIDENCY").ok(),
        qprompt.len(),
        tprompt.len()
    );
    let qf =
        |t: &[u32], c: &mut KvCache, x: &mut InferenceContext, s: &mut Option<DecodeSession>| {
            qwen.forward_with_kv_context_persistent(t, c, x, s)
                .map_err(|e| format!("{e:?}"))
        };
    let tf =
        |t: &[u32], c: &mut KvCache, x: &mut InferenceContext, s: &mut Option<DecodeSession>| {
            tiny.inner()
                .forward_with_kv_context_persistent(t, c, x, s)
                .map_err(|e| format!("{e:?}"))
        };
    // One warm-up visit of each so kernel and pipeline compilation is not in the timed visits.
    let _ = visit(&dev, qshape, &qprompt, 2, &qf);
    let _ = visit(&dev, tshape, &tprompt, 2, &tf);
    dev.clear_weight_residency();
    std::thread::sleep(std::time::Duration::from_millis(300));
    println!(
        "VRAM MiB after warm-up and clear: {}",
        vram_used_mib().unwrap_or(0)
    );
    for r in 0..rounds {
        for which in 0..2 {
            let (name, (hashes, pf, fd)) = if which == 0 {
                ("qwen3", visit(&dev, qshape, &qprompt, 3, &qf))
            } else {
                ("tinyllama", visit(&dev, tshape, &tprompt, 3, &tf))
            };
            std::thread::sleep(std::time::Duration::from_millis(200));
            let hs: Vec<String> = hashes.iter().map(|h| format!("{h:016x}")).collect();
            println!(
                "VISIT round {r} {name}: prefill {pf:.0} ms, first decode {fd:.0} ms; hashes {}; VRAM MiB now {}; {:?}",
                hs.join(","),
                vram_used_mib().unwrap_or(0),
                dev.weight_residency_stats()
            );
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = sampler.join();
    let end = vram_used_mib().unwrap_or(0);
    dev.clear_weight_residency();
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!(
        "SWAP VRAM MiB: baseline {base}, peak {}, end (device alive) {end}, after clear {}",
        peak.load(std::sync::atomic::Ordering::Relaxed),
        vram_used_mib().unwrap_or(0)
    );
}
