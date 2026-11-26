# Row 175 (provisional): GGUF encoder embeddings (design note)

Status: DESIGN, nothing built. Opened 2026-11-26. Owner: Fuel 1. Consumer: Lightbulb `/v1/embeddings` (CireSnave board 505: "Yes for the embeddings endpoint."). Scheduled, not a skip (6.21). The row number is provisional; the PM renumbers at the gate.

## Facts at origin/main dcf8c46a (read by `git show`, 2026-11-26)
- Encoder models exist in `fuel-transformers/src/models/`: `lazy_bert`, `lazy_distilbert`, `lazy_jina_bert`, `lazy_nomic_bert`, `lazy_modernbert`, `lazy_debertav2`, `lazy_xlm_roberta`, `lazy_stella_v5`, `lazy_nvembed_v2`.
- Every `forward` takes ONE sequence (`&[u32]`, batch 1). `lazy_bert::forward` documents "Assumes all tokens are valid (no padding / mask)". `nomic_bert`, `jina_bert`, `modernbert` take an optional additive `attention_mask` Tensor; `stella_v5::forward_with_mask` and `nvembed_v2::forward` take `&[u32]` masks.
- Weights: safetensors via `MmapedSafetensors::load_from_mmapped` (read for bert, jina_bert, modernbert).
- GGUF: 0 hits for "gguf" in the encoder model files; the same search finds it in `lazy_quantized_qwen2.rs` (positive control). No GGUF encoder path.
- Pooling: per-model code (mean in stella/nvembed/bert example, last-token in the gte-qwen example); `Tensor::l2_normalize` in `fuel-tensor/src/lazy.rs`. No shared API.

## Goal
One shared embedding API that loads the GGUF encoders Ollama users run and returns L2-normalised vectors for padded batches, with reference vectors proving agreement.

## UNVERIFIED (from memory of llama.cpp conventions; slice 0 reads real GGUF headers and cites each key)
- Architecture strings: bert-family (all-minilm, bge-large, mxbai-embed-large, snowflake-arctic-embed) `bert`; nomic-embed-text `nomic-bert`; bge-m3 an XLM-RoBERTa checkpoint, probably `bert` with a different tokenizer model. Keys expected: `<arch>.pooling_type` (none/mean/cls/last), `<arch>.attention.causal`, `<arch>.embedding_length`, `<arch>.block_count`, `<arch>.attention.layer_norm_epsilon`, `<arch>.context_length`, `tokenizer.ggml.model`.
- Slice 0 deliverable: a table (family, file, sha256, arch string, pooling_type, causal, tokenizer model) read with Fuel's own GGUF reader, plus the origin and licence of each file used.

## Design
1. `EmbeddingModel` trait in a new module (location decided with Lightbulb's note): `embed(&self, batch: &[Vec<u32>]) -> Result<Tensor /* [B, hidden], L2-normalised */>`, `pooling() -> Pooling { Mean, Cls, LastToken }`, `hidden_size()`. Pooling comes from GGUF `pooling_type` (or the safetensors config), never hard-coded per call.
2. Padding: right-pad to the batch max, build a `[B, 1, 1, S]` additive mask (0 / -inf as a select, not an additive band: see PIP #715 M1), bidirectional attention (no causal mask). Mean pooling divides by the mask sum; CLS takes position 0; last-token takes the last unmasked position. A fully-masked row is a typed Err, not a NaN.
3. Encoders get a `batch` dimension: today's batch-1 forwards become the `B=1` case of the new path; the old signatures stay until callers move.
4. GGUF encoder loader: maps GGUF tensor names to the existing weight structs, reusing the quantized-matmul path for quantized files.

## Correctness gate
Per family, reference vectors from llama.cpp (`llama-embedding`) and sentence-transformers on a fixed 8-sentence corpus including a padded-length mix, recorded with tool version, model file sha256 and licence. Pass = cosine >= tolerance calibrated by sabotage (not inherited), f32 and f16, CPU first, then CUDA, then Vulkan. A batch of N padded sequences must equal N singleton runs within that tolerance (the padding-invariance control).

## Lightbulb requirements (their finding 2026-11-26, read from their message, not their note)
- Token ids only; Lightbulb tokenizes, counts usage and rejects over-length input (never truncates). Ids INCLUDE the model's special tokens (CLS/SEP); the mask covers them.
- `max_position()` exposed beside `hidden_size()`.
- `normalized()` flag from config, not hard-coded L2 (some models want raw vectors).
- Matryoshka: nomic-embed-text-v1.5 applies layer_norm over the full width BEFORE truncating, then L2; truncate-then-renorm is wrong for it and fine for mxbai. Expose an optional `layer_norm + truncate(dim) + L2` step (or the pre-L2 pooled vector), selected by model config.
- Call shape: one request array at a time, B <= 64. First model: nomic-embed-text (nomic_bert) or bge/MiniLM, whichever lands first; slice 1 stays bert-family MiniLM/bge, nomic-bert is slice 3 unless Lightbulb asks to swap.

## Slices
0. GGUF header table (above) + gaps/ROADMAP citation. 1. ONE family end to end: bert-family MiniLM/bge on CPU f32 (smallest): loader, mask, pooling, normalise, reference vectors. 2. f16, then CUDA/Vulkan. 3. nomic-bert, then xlm-roberta (bge-m3), then arctic. 4. Lightbulb integration; `release_pooled_memory` for encoders when it lands (Lightbulb's ask, not built).

— lane: fuel
