// Throwaway: lightbulb's real prefill route -- PagedSessionScheduler, time to first token.
use fuel::lazy::SamplingStrategy;
use fuel::lazy_llama_full::LlamaFullConfig;
use fuel::lazy_quantized_llama::QuantizedLlama3Model;
use fuel_inference::multi_session::{KvBudget, PagedSessionScheduler};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::env::var("GGUF")?;
    let ntok: usize = std::env::var("NTOK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(145);
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
    let mut s = PagedSessionScheduler::new(
        &model,
        KvBudget {
            block_size: 16,
            num_blocks: (ntok + 64) / 16 + 4,
        },
        fuel_ir::DType::F32,
        &fuel_examples::device(std::env::var("DEVICE").map(|v| v == "cpu").unwrap_or(true))?,
    )?;
    let prompt: Vec<u32> = (0..ntok as u32).map(|i| 100 + (i * 7) % 900).collect();
    let id = s.add_session(&prompt, SamplingStrategy::Greedy, None, 4)?;
    fuel_dispatch::compiled::op_profile::set_enabled(true);
    let t = Instant::now();
    let _ = s.step();
    let ttft = t.elapsed().as_secs_f64();
    let first = s
        .session_new_tokens(id)
        .map(|t| t.to_vec())
        .unwrap_or_default();
    for (name, n, dd) in fuel_dispatch::compiled::op_profile::take_phases()
        .iter()
        .take(8)
    {
        println!("PHASE {name} n={n} {:.3}s", dd.as_secs_f64());
    }
    println!(
        "OPTIMIZE_CALLS {}",
        fuel::pipelined_bridge::optimize_calls_thread_local()
    );
    println!("SCHED tokens={ntok} ttft={ttft:.2}s first_tokens={first:?}");
    Ok(())
}
