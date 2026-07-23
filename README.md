# HandmadeLLM

A modern, Llama-style LLM built **entirely from scratch** — tokenizer, model, training loop, everything. No Hugging Face `Trainer`, no imported tokenizers. Trained, evaluated, quantized and served on a single RTX 5070 Ti (12 GB).

> **Status:** Phases 0–1 complete (core implementation + verified training stack). Phases 2–8 in progress — see [Roadmap](#roadmap).

## What's implemented

- **Byte-level BPE tokenizer from scratch** ([llm/bpe.py](llm/bpe.py)) — merge training with incremental pair counts, encode/decode, save/load. Fully tested, including UTF-8/emoji roundtrips.
- **Llama-style transformer** ([llm/model.py](llm/model.py)):
  - RoPE (rotary position embeddings)
  - RMSNorm (pre-norm)
  - Causal attention with **GQA** (grouped-query attention)
  - SwiGLU MLP, weight tying, GPT-2-style scaled residual init
  - **KV cache** for generation, with tests proving cached ≡ uncached greedy decoding
- **Hand-written training loop** ([llm/train.py](llm/train.py)) — AdamW with decay/no-decay groups, bf16 autocast, gradient accumulation, gradient clipping, cosine LR with warmup, gradient checkpointing, `torch.compile`, checkpoint/resume, JSONL metrics + optional W&B.
- **Data pipeline** ([scripts/prepare_tinystories.py](scripts/prepare_tinystories.py)) — TinyStories → own tokenizer → uint16 memmap bins, multiprocess tokenization.
- **Test suite** ([tests/](tests/)) — causality, KV-cache equivalence, RoPE relative-position property, GQA shapes, BPE roundtrips, overfit smoke test.

## Verified on hardware

Phase 0 definition-of-done (RTX 5070 Ti Laptop, Blackwell `sm_120`, torch 2.11 cu128):

| Check | Result |
|---|---|
| `torch.cuda.is_available()` | ✅ torch 2.11.0+cu128, capability `(12, 0)` |
| Overfit one batch, 9.8M model, 500 steps | loss **5.51 → 0.0043** in 17.3 s |
| Throughput (bf16, batch 32×256) | **237k tok/s** |
| Peak VRAM | 1.29 GB |
| Test suite | 22/22 passing |

Reproduce with `python scripts/overfit_sanity.py`.

## Quickstart

```bash
# 1. Environment (Blackwell needs the cu128 wheels)
python -m venv .venv && .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. Sanity check: overfit one batch (proves the whole stack works)
python scripts/overfit_sanity.py

# 3. Data + tokenizer (TinyStories, ~2 GB download)
python scripts/prepare_tinystories.py --vocab-size 8192

# 4. Train the 30M model
python -m llm.train --config configs/tinystories_30m.yaml

# 5. Generate
python -m llm.sample --ckpt checkpoints/tinystories_30m/best.pt --prompt "Once upon a time"
```

Run tests: `pytest tests/ -v`

## Architecture

| | 30M ([config](configs/tinystories_30m.yaml)) | Flagship 113M ([config](configs/flagship_113m.yaml)) |
|---|---|---|
| dim / layers | 512 / 8 | 768 / 16 |
| heads (q / kv) | 8 / 4 | 12 / 4 |
| context | 512 | 1024 |
| vocab (own BPE) | 8,192 | 16,384 |
| effective batch | 131k tok | 524k tok |

## Roadmap

- [x] **Phase 0** — env de-risk (Blackwell + cu128), repo scaffold, overfit sanity ✅
- [x] **Phase 1** — BPE tokenizer, Llama-style model, training loop, tests ✅
- [ ] **Phase 2** — efficiency study: before/after table (bf16, accum, checkpointing, flash SDPA, 8-bit optim)
- [ ] **Phase 3** — flagship training + mini scaling-law study (10M→113M, fixed compute)
- [ ] **Phase 4** — KV-cache benchmarks, int8/int4 quantization, FastAPI streaming endpoint, serving benchmark
- [ ] **Phase 5** — SFT instruction tuning (+ DPO stretch)
- [ ] **Phase 6** — eval harness: perplexity, downstream benchmark, ablations (RoPE vs learned, GQA vs MHA, SwiGLU vs GELU)
- [ ] **Phase 7** — fused Triton kernel (RMSNorm) + correctness test + benchmark
- [ ] **Phase 8** — technical writeup

## Design decisions

- **GQA over MHA**: 4 KV heads for 12 query heads cuts KV-cache memory ~3× at inference, nearly free in quality at this scale — and it's what 2024/25 production models actually do.
- **Byte-level BPE**: no out-of-vocabulary tokens ever; the tokenizer is ~200 lines and auditable. Training uses incremental pair counting (only words containing the merged pair are touched), so an 8k vocab trains in minutes in pure Python.
- **SDPA for attention**: `F.scaled_dot_product_attention` gives flash/mem-efficient kernels today; a hand-written Triton kernel replaces RMSNorm later as a like-for-like benchmark (Phase 7).
- **uint16 memmap data**: the whole dataset never touches RAM; random-offset sampling keeps the loader at ~zero cost vs the GPU step.

## License

MIT
