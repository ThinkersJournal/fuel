// Throwaway: per-token cost of lightbulb's PAGED prefill (feeds the prompt one token at a time).
use fuel::lazy_llama_full::LlamaFullConfig;
use fuel::lazy_quantized_llama::QuantizedLlama3Model;
use fuel_inference::multi_session::{DecodeModel, PagedDecodeModel};
use fuel_kv_pool::KvGeometry;
use fuel_tensor::Device;
use fuel_tensor::inference_context::{PagedDecodePlan, PagedDecodeSession};
use fuel_tensor::kv_block_pool_device::DeviceKvPool;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::var("GGUF")?;
    let ntok: usize = std::env::var("NTOK").ok().and_then(|v| v.parse().ok()).unwrap_or(145);
    let plan_once = std::env::var("REPLAN").is_err();
    let content = fuel_loaders::quantized::gguf_file::Content::open(std::path::Path::new(&path))?;
    let d = fuel_loaders::quantized::config_from_gguf::derive_config(
        &content,
        fuel_loaders::quantized::arch::Architecture::Llama,
    )?;
    let cfg = LlamaFullConfig {
        hidden_size: d.hidden_size,
        intermediate_size: d.intermediate_size,
        vocab_size: d.vocab_size,
        num_hidden_layers: d.n_layers,
        num_attention_heads: d.n_heads,
        num_key_value_heads: d.n_kv_heads,
        head_dim: d.hidden_size / d.n_heads.max(1),
        rms_norm_eps: d.rms_norm_eps,
        rope_theta: d.rope_theta,
        max_position_embeddings: d.max_position_embeddings,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        tie_word_embeddings: false,
    };
    let model = QuantizedLlama3Model::from_gguf(&path, &cfg)?;
    let dev = Device::cpu();
    let block = 16usize;
    let geom = KvGeometry {
        n_layers: DecodeModel::n_layers(&model),
        num_blocks: (ntok + 32) / block + 4,
        block_size: block,
        n_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        elem_size: 4,
    };
    let mut pool = DeviceKvPool::new(geom, fuel_ir::DType::F32, &dev)?;
    let session = pool.core_mut().open();
    let cap = (ntok + 32).div_ceil(block);
    let mut ds: Option<PagedDecodeSession> = None;
    if std::env::var("FUEL_PREFILL_PROFILE").is_ok() {
        fuel_dispatch::compiled::op_profile::set_enabled(true);
    }
    let tokens: Vec<u32> = (0..ntok as u32).map(|i| 100 + (i * 7) % 900).collect();
    let t0 = Instant::now();
    let mut per = Vec::new();
    for &tok in &tokens {
        let t = Instant::now();
        if plan_once {
            PagedDecodeModel::forward_paged_step_persistent(&model, tok, &mut pool, session, cap, PagedDecodePlan::PlanOnce, &mut ds)?;
        } else {
            PagedDecodeModel::forward_paged_step(&model, tok, &mut pool, session)?;
        }
        per.push(t.elapsed().as_secs_f64());
    }
    let total = t0.elapsed().as_secs_f64();
    let first = per[0];
    let mut rest = per[1..].to_vec();
    rest.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("PAGED plan_once={plan_once} tokens={ntok} total={total:.2}s first_step={first:.3}s median_step={:.3}s max_step={:.3}s", rest[rest.len()/2], rest[rest.len()-1]);
    for (name, n, dd) in fuel_dispatch::compiled::op_profile::take_phases().iter().take(6) {
        println!("PHASE {name} n={n} {:.3}s", dd.as_secs_f64());
    }
    for (op, be, n, dd) in fuel_dispatch::compiled::op_profile::take().iter().take(4) {
        println!("OP {op} {be} n={n} {:.3}s", dd.as_secs_f64());
    }
    Ok(())
}
