// Throwaway TTFT probe (not committed): lightbulb's prefill path on a GGUF, CPU.
use fuel::lazy_llama_full::LlamaFullConfig;
use fuel::lazy_quantized_llama::QuantizedLlama3Model;
use fuel_inference::multi_session::DecodeModel;
use fuel_tensor::Device;
use fuel_tensor::inference_context::{InferenceContext, KvCache};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::var("GGUF")?;
    let ntok: usize = std::env::var("NTOK").ok().and_then(|v| v.parse().ok()).unwrap_or(140);
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
    let tokens: Vec<u32> = (0..ntok as u32).map(|i| 100 + (i * 7) % 900).collect();
    let dev = Device::cpu();
    let specs = DecodeModel::layer_state_specs(&model);
    let (n_kv, hd) = match specs[0] {
        fuel_inference::decode_state_spec::LayerStateSpec::KeyValue { n_kv_heads, head_dim } => (n_kv_heads, head_dim),
        _ => anyhow::bail!("not KV"),
    };
    let mut cache = KvCache::with_capacity(
        DecodeModel::n_layers(&model), n_kv, hd, ntok + 8, fuel_ir::DType::F32, &dev)?;
    let mut ctx = InferenceContext::new(dev);
    let mut session = None;
    if std::env::var("FUEL_PREFILL_PROFILE").is_ok() {
        fuel_dispatch::compiled::op_profile::set_enabled(true);
        fuel_dispatch::compiled::op_profile::set_trace(true);
    }
    let t = Instant::now();
    let logits = DecodeModel::forward_with_kv_context_persistent(&model, &tokens, &mut cache, &mut ctx, &mut session)?;
    println!("PREFILL tokens={ntok} wall={:.3}s logits_len={} first={:.6}", t.elapsed().as_secs_f64(), logits.len(), logits[0]);
    let mut tr = fuel_dispatch::compiled::op_profile::take_trace();
    for r in tr.iter().rev().filter(|r| r.op.contains("MatMul")).take(2) { println!("LASTLAUNCH {} in_bytes={} {:.3}s", r.op, r.bytes, r.dt.as_secs_f64()); }
    tr.sort_by_key(|r| std::cmp::Reverse(r.dt));
    for r in tr.iter().take(4) { println!("LAUNCH {} in_bytes={} {:.3}s", r.op, r.bytes, r.dt.as_secs_f64()); }
    for (op, be, n, dd) in fuel_dispatch::compiled::op_profile::take().iter().take(4) {
        println!("OP {op} {be} n={n} {:.3}s", dd.as_secs_f64());
    }
    Ok(())
}
