# Row 173 (provisional; first drafted as 175, the file name keeps that): GGUF encoder embeddings (design note)

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

## Slice 0, partial: two real GGUF headers read (2026-11-26)
Source: llama.cpp `models/ggml-vocab-*.gguf` fixtures in the box's GGUF corpus (`C:\models\gguf-corpus/llamacpp-vocab/`, listed with sha256 in its `MANIFEST.json`; llama.cpp is MIT). They are VOCAB-ONLY (0 tensors): they carry metadata, not weights. Read with a throwaway GGUF v3 key reader (`/c/cb/ggufmeta.py`, not in the repo).
- `ggml-vocab-bert-bge.gguf` (sha256 `fbcbe222...54a854`, 627,549 bytes): `general.architecture = bert`; `bert.block_count 12`, `bert.context_length 512`, `bert.embedding_length 384`, `bert.feed_forward_length 1536`, `bert.attention.head_count 12`, `bert.attention.layer_norm_epsilon 1e-12`, `bert.attention.causal = False`, `tokenizer.ggml.model = bert`, `tokenizer.ggml.pre = bert-bge`, `cls_token_id 101`, `seperator_token_id 102` (llama.cpp's spelling), `padding_token_id 0`, `token_type_count 2`. NO `bert.pooling_type` key in this fixture: a real bge/MiniLM GGUF must be read to learn whether the converter writes it (UNVERIFIED for weights files).
- `ggml-vocab-nomic-bert-moe.gguf` (sha256 `90a67469...fe5f9`, 6,821,877 bytes): `general.architecture = nomic-bert-moe` (the MoE variant, not v1/v1.5 `nomic-bert`), `nomic-bert-moe.pooling_type = 1` (mean), `.attention.causal = False`, `.rope.freq_base 10000`, `.moe_every_n_layers 2`, `.expert_count 8`, `.expert_used_count 2`, `tokenizer.ggml.model = t5` (a SentencePiece unigram tokenizer, 250,048 tokens).
- Established: the bert family uses arch string `bert` with `<arch>.attention.causal = false`; `<arch>.pooling_type` exists as a key (1 = mean in this file). NOT established: the arch strings of nomic-embed-text v1/v1.5, mxbai, all-minilm, bge-m3, arctic (no file of those on the box); the pooling_type encoding for cls/last (assumed 2/3 from llama.cpp, unread); tensor names (no weights read). Slice 0 stays open until a real weights GGUF per family is read.

## Slice 0 result: three real weights GGUF headers read (2026-11-27)
Method: the first 4,000,000 bytes of each file by HTTP range from huggingface.co (the metadata and tensor-info block precede the weights, so no weights were downloaded; the files are not hashed because a prefix has no file hash), parsed with a bounded GGUF v3 reader (`/c/cb/ggufmeta2.py`, not in the repo, run `python -I` on the untrusted bytes). Repo revisions: `CompendiumLabs/bge-small-en-v1.5-gguf@d32f8c040e` (MIT, `bge-small-en-v1.5-f16.gguf`, 67,308,128 bytes), `second-state/All-MiniLM-L6-v2-Embedding-GGUF@544f204f2e` (Apache-2.0, `all-MiniLM-L6-v2-Q4_0.gguf`, 19,699,648 bytes), `nomic-ai/nomic-embed-text-v1.5-GGUF@0188c9bf40` (Apache-2.0, `nomic-embed-text-v1.5.Q4_0.gguf`, 77,802,880 bytes); licences as the repo metadata states them, the model cards not read.

| key | bge-small-en-v1.5 (f16) | all-MiniLM-L6-v2 (Q4_0) | nomic-embed-text-v1.5 (Q4_0) |
|---|---|---|---|
| `general.architecture` | `bert` | `bert` | `nomic-bert` |
| blocks / hidden / ffn / heads | 12 / 384 / 1536 / 12 | 6 / 384 / 1536 / 12 | 12 / 768 / 3072 / 12 |
| `<arch>.context_length` | 512 | 512 | 2048 |
| `<arch>.attention.causal` | False | False | False |
| `<arch>.pooling_type` | **2** | **1** | **1** |
| `<arch>.attention.layer_norm_epsilon` | 1e-12 | 1e-12 | 1e-12 |
| `<arch>.rope.freq_base` | (absent) | (absent) | 1000.0 |
| `tokenizer.ggml.model` | `bert` (30,522-row embedding) | `bert` | `bert` |
| special ids | cls 101, sep 102 (`seperator_token_id`), pad 0, mask 103, unk 100 | same | same, no `cls_token_id` key |
| tensors | `position_embd.weight`, `token_types.weight [384,2]`, `token_embd_norm.*`, per block `attn_{q,k,v}.{weight,bias}`, `attn_output(.norm).*`, `ffn_up/down.*`, `layer_output_norm.*` | same names | NO `position_embd` (rotary), fused `attn_qkv.weight [768,2304]`, `ffn_gate.weight` (gated FFN), no biases on the projections seen |

What this settles (each from a file above, not from memory): arch strings `bert` and `nomic-bert`; the `pooling_type` encoding 1 = mean (MiniLM, nomic) and 2 = CLS (bge) as written by the converter, so the loader MUST read the key rather than assume mean; `attention.causal = False` everywhere; the bert tokenizer is WordPiece (`tokenizer.ggml.model = bert`), so a WordPiece tokenizer is on slice 1's path (the sentencepiece unigram of `nomic-bert-moe` is a later family); nomic needs rotary position handling, a fused qkv split and a gated FFN that the plain bert path does not. Tensor dtype ids seen: 0 = F32, 1 = F16, 2 = Q4_0, 8 = Q8_0. NOT settled: mxbai, arctic, bge-m3 (not read); whether `token_embd_norm` is applied before the position add exactly as HF BERT (a numeric reference run decides, slice 1); the Matryoshka behaviour of nomic (a model-card fact, not a GGUF key).

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
