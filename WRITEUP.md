# Building a modern LLM from scratch on one laptop GPU

*A Llama-style decoder-only transformer — tokenizer, model, training, quantization, serving, evaluation, and a fused Triton kernel — written by hand and run end-to-end on a single RTX 5070 Ti (12 GB, Blackwell).*

This is the honest engineering log: what I built, the numbers I measured, and what I'd do differently. The signal here isn't model size — it's depth, correctness, and reproducibility. Every number below is reproducible from the repo.

---

## 1. Why this, and what "from scratch" means

No Hugging Face `Trainer`, no imported tokenizer. I wrote the BPE merge algorithm, the RoPE/RMSNorm/GQA/SwiGLU transformer, the training loop, the KV cache, weight-only quantization, and a fused Triton RMSNorm kernel. The one thing I *didn't* reinvent is the attention inner loop — I use `F.scaled_dot_product_attention` so I get FlashAttention-class kernels — and I later replace RMSNorm with my own Triton kernel as a like-for-like comparison.

The target architecture is deliberately 2024/25, not GPT-2-2019: rotary positions, RMSNorm pre-norm, grouped-query attention, SwiGLU MLP, weight tying.

## 2. The architecture

| choice | what | why |
|---|---|---|
| **RoPE** | rotary position embedding | relative positions, extrapolates better than learned, no position params |
| **RMSNorm** | pre-norm, no mean-centering | cheaper than LayerNorm, stable, the modern default |
| **GQA** | 12 query heads share 4 KV heads | ~3× smaller KV cache at inference for ~free quality |
| **SwiGLU** | gated MLP, `w2(silu(w1 x) * w3 x)` | consistently beats plain GELU per parameter |
| **weight tying** | `lm_head.weight = tok_emb.weight` | saves `vocab×dim` params, helps small models |

The ablation table (§7) puts numbers on the last three instead of just asserting them.

## 3. Tokenizer: BPE from scratch

Byte-level BPE with the classic word-frequency + incremental pair-count formulation: pre-tokenize with a regex, keep unique "words" as byte sequences, and on each merge only touch words that actually contain the merged pair. That's what makes it fast in pure Python — an 8k-vocab tokenizer trains on 31 MB of TinyStories in **11 seconds**. Byte-level means there is never an out-of-vocabulary token; UTF-8 and emoji round-trip exactly (tested).

## 4. De-risking the environment first (Phase 0)

The #1 bottleneck of a Blackwell card isn't compute, it's the driver/PyTorch stack. `sm_120` needs the CUDA 12.8 wheels (`pip install torch --index-url .../cu128`), and `torch.compile` needs Triton — which on Windows means `triton-windows`. I front-loaded all of this, then proved the training stack works before scaling: a 9.8M model **overfits one batch to loss 0.004** in 17 s. If that hadn't collapsed, something was broken and I'd want to know at minute 10, not hour 3.

## 5. The 30M result (Phase 1)

A 28M-param model (dim 512, 8 layers, GQA 8q/4kv, ctx 512, own 8k BPE), trained ~2.6 h on TinyStories:

- final val loss **1.24**, held-out **perplexity 3.75**
- generates coherent stories with dialogue, narrative arc, even "moral of the story" endings (`samples/tinystories_30m.md`)

Honest failure modes: occasional logical slips ("a small box. It was a very big box"), and it sometimes trails off at the token budget. Expected at 28M — the point is coherence, not a production assistant.

### The flagship (113M)

The headline model: dim 768, 16 layers, GQA 12q/4kv, **1024 context, 16k BPE**, 113M params. It only fits and trains at a reasonable speed *because* of the Phase 2 efficiency work — bf16 + gradient checkpointing + compile keep it at **5.3 GB peak** and ~34k tok/s, so 4000 steps (~655M tokens) finished in ~5 h. Final val loss 1.26, perplexity 3.71. Qualitatively it's a clear step up: named characters, richer vocabulary, longer arcs (`samples/flagship_113m.md`).

An honest measurement note: the flagship's per-token val loss (1.26) is **not** comparable to the 30M's (1.24) — they use different tokenizers, and a 16k vocab packs more information per token, so per-token cross-entropy is naturally higher. Bits-per-byte would be the fair cross-tokenizer metric; I report both models' numbers but don't pretend the raw losses rank them.

## 6. Making 12 GB enough (Phase 2)

This is the part that's actually ML *engineering*. The flagship 113M config, each optimization applied cumulatively:

| setup | tokens/s | peak VRAM |
|---|---|---|
| fp32 baseline | 8,893 | 12.39 GB (≈ card limit) |
| + bf16 autocast | 28,498 | 9.33 GB |
| + gradient checkpointing | 23,313 | **3.95 GB** |
| + 3× bigger micro-batch | 22,215 | 8.10 GB |
| + torch.compile | **34,496** | 6.03 GB |

Takeaways: bf16 is a **3.2× throughput** win on Blackwell and the single biggest lever. Gradient checkpointing trades ~15% compute for a **3× VRAM cut** (12.4 → 4 GB) — which is what actually lets a 113M model with a 1024 context fit, and then run at a 3× larger micro-batch. `torch.compile` recovers the checkpointing overhead and then some.

## 7. Inference: KV cache + quantization (Phase 4)

**KV cache** — flagship, 512-token generation: **1.94×** faster (156 vs 80 tok/s), and the gap widens with context length because uncached decoding is O(T²) in the sequence. Correctness is pinned by a test asserting cached and uncached greedy decoding produce *identical* tokens.

**Weight-only quantization from scratch** (per-row symmetric, int4 bit-packed two-per-byte), measured on the trained 30M:

| precision | linear weights | perplexity | vs fp |
|---|---|---|---|
| fp | 111 MB | 3.42 | — |
| int8 | 40 MB | 3.42 | 2.7× smaller, **+0.0%** |
| int4 | 29 MB | 3.61 | 3.9× smaller, +5.5% |

int8 is a free memory win; int4 costs ~5% perplexity for another ~30% off. I kept `lm_head` (tied to the embedding) in full precision. Honest caveat: my forward is dequant-then-bf16-matmul, so it doesn't *speed up* inference — a fused INT8 GEMM is future work; what I'm measuring is the accuracy/size trade-off of the scheme.

## 8. A fused Triton kernel (Phase 7)

RMSNorm reads its input several times over HBM in the naive PyTorch version. The fused kernel loads each row once into SRAM, computes the norm, and writes the result — one launch per row-block, forward **and** backward (the backward fuses the input-gradient and accumulates the weight-gradient with atomics). Correctness is asserted against the PyTorch reference (fp32 and bf16, fwd and bwd). Speedup at dim 768, bf16:

| tokens | torch fwd | triton fwd | speedup |
|---|---|---|---|
| 16,384 | 1.067 ms | 0.071 ms | **15.0×** |
| 262,144 | 14.736 ms | 1.647 ms | 9.0× |

## 9. Scaling study (Phase 3)

Four sizes, **identical ~110M-token budget** (data-controlled), TinyStories, same LR schedule:

| model | dim×layers | non-emb params | final val loss |
|---|---|---|---|
| tiny | 256×4 | 3.0M | 1.998 |
| small | 384×6 | 9.7M | 1.739 |
| med | 512×8 | 23.6M | 1.615 |
| large | 640×10 | 45.5M | 1.540 |

Fitting a power law gives **L ≈ 2.197 · N^(−0.096)** with **R² = 0.987** — a clean, monotone Chinchilla-style curve over a 15× parameter range (`samples/scaling.svg`). The diminishing returns are visible: the first 3× of params buys −0.26 loss, the next 4.7× only −0.20. Honest scope note: this is *data-controlled* (fixed tokens), not a full compute-optimal sweep, so it shows the shape, not the optimal token/param allocation.

## 10. Ablations (Phase 6)

One architectural change at a time vs the modern baseline (RoPE + SwiGLU + RMSNorm + GQA), same fixed budget on TinyStories:

| change | final val loss | Δ vs baseline |
|---|---|---|
| — (baseline) | 1.813 | — |
| RoPE → learned pos-emb | 1.891 | **+0.078** |
| SwiGLU → GELU MLP | 1.860 | **+0.047** |
| RMSNorm → LayerNorm | 1.810 | −0.003 |
| GQA → full MHA | 1.798 | −0.014 |

What the numbers actually say — and where they refuse to flatter the defaults:

- **RoPE earns its place** (+0.078 without it): the single biggest architectural lever here.
- **SwiGLU earns its place** (+0.047 without it), at matched parameter count.
- **RMSNorm is a wash on quality** (−0.003) — so I keep it for being *cheaper*, not more accurate. Honest: at this scale LayerNorm would lose nothing in loss.
- **GQA costs a hair of loss** (MHA is 0.014 better) but buys a **3× smaller KV cache**. That's the trade I'd make every time for inference, and it's why production models do too — but the ablation is honest that it *is* a (tiny) trade, not a free win.

## 10b. Instruction tuning (Phase 5)

SFT turns "generates text" into "follows instructions." I fine-tuned the 30M base on 30k TinyStories-Instruct `(prompt, response)` pairs, with the loss **masked over the prompt** — only the response tokens are supervised (right-padded batches; causal attention means real tokens never see the trailing pad, so no mask surgery). 3 epochs, lr 2e-5, SFT val loss 1.14.

Prompt format: `Features:` / `Words:` (the story must use them) / `Summary:` (the plot to follow) / `Story:`.

- **Before** (base model): treats the fields as noise — rambles, ignores the required words, leaks literal `Story:`/`Apparent:` tokens.
- **After** (SFT): reliably incorporates every required word and matches the summary. Given *dog, jump, happy* it writes a story about a happy dog that jumps; given *boat, river, brave* it writes a brave girl sailing a boat down a river.

Full before/after in `samples/sft_before_after.md`. DPO (preference alignment) is the natural next stretch and is scoped but not yet built.

## 11. What didn't work / what I'd do differently

- **`torch.compile` on Windows** silently needs Triton; the first run failed with a cryptic `TritonMissing`. I made compile degrade gracefully to eager and documented `triton-windows`. Lesson: probe compilation *before* the training loop so it fails in 5 s, not mid-run.
- **Quantization speed**: I built the accuracy/size story honestly but a dequant→matmul path is not faster. Doing it "for real" means an INT8 GEMM (CUTLASS/Triton) — scoped as future work rather than faked.
- **Real-language pretraining** (FineWeb-Edu) is deliberately out of scope: trillions of tokens don't fit on a laptop and chasing it would have burned the whole budget for a worse story than a finished TinyStories model.
- If I restarted: wire W&B from day 1 (I logged to JSONL and added W&B as opt-in), and build the eval harness *before* the flagship run so every checkpoint is scored automatically.

## 12. Reproducing

Everything is in the README quickstart. Configs are versioned per experiment, checkpoints carry their config, and the results tables regenerate from the `scripts/bench_*.py` and `scripts/*_study.py` drivers. Clone, read, and — if you want — run.
