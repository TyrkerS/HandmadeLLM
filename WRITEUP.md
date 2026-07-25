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

An honest measurement note: the flagship's per-token val loss (1.26) is **not** comparable to the 30M's (1.24) — they use different tokenizers, and a 16k vocab packs more information per token, so per-token cross-entropy is naturally higher. The fair cross-tokenizer metric is **bits-per-byte** (total NLL ÷ UTF-8 bytes), which I compute directly ([llm/eval.py](llm/eval.py)) — and it drove a real iteration on this project:

| model | tokens seen | val bits-per-byte (↓) |
|---|---|---|
| 30M (8k vocab) | ~1.5B | 0.4556 |
| flagship — first run | 655M | 0.4537 |
| **flagship — longer run** | ~984M | **0.4491** |

The first flagship beat the 30M by only 0.4% for 4× the parameters — the tell of an **under-trained** model (655M tokens ≈ 5.8 tok/param vs Chinchilla-optimal ~20). So I re-ran it longer (resuming from a checkpoint — the training loop supports it). The gap over the 30M widened from 0.4% to **1.4%** (val 1.257 → 1.232, ppl 3.75 → 3.70), and generation quality visibly improved (multi-beat plots, richer vocabulary).

The honest twist: val **plateaued** around 1.23 in the final 500 steps. TinyStories is deliberately simple, so a 113M model saturates well before a full Chinchilla token budget — the original diagnosis ("under-trained") was right, but the real ceiling here is the *dataset*, not the compute. A harder corpus (FineWeb-Edu) is where more parameters would keep paying off; on TinyStories, ~1.23 val is about the floor. Catching this required bits-per-byte — measuring it is what turned "the headline model is my weakest point" into a documented, fixed iteration.

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

**Continuous batching** ([llm/engine.py](llm/engine.py)) — the serving trick behind vLLM. A naive server decodes one request at a time; my engine keeps a dynamic batch, advances every in-flight request per forward pass, and admits new arrivals mid-flight. The hard parts: ragged sequence lengths (left-padded KV cache + a key-padding mask so queries never see pad slots) and per-sequence RoPE positions (each row is at a different absolute position, so I thread `position_ids` through instead of a shared `start_pos`). Correctness is pinned to sequential greedy decoding token-for-token. Aggregate throughput on the flagship: sequential serving is flat at ~109 tok/s regardless of load, while batching scales to **634 tok/s at 16 concurrent requests (5.8×)** before the GPU saturates. Honest: at concurrency 1 the engine's per-step overhead makes it 0.78× — batching is a throughput-under-load win, not a single-request one.

**Weight-only quantization from scratch** (per-row symmetric, int4 bit-packed two-per-byte), measured on the trained 30M:

| precision | linear weights | perplexity | vs fp |
|---|---|---|---|
| fp | 111 MB | 3.42 | — |
| int8 | 40 MB | 3.42 | 2.7× smaller, **+0.0%** |
| int4 | 29 MB | 3.61 | 3.9× smaller, +5.5% |

int8 is a free memory win; int4 costs ~5% perplexity for another ~30% off. I kept `lm_head` (tied to the embedding) in full precision.

That table quantizes weights but dequantizes to bf16 for the matmul, so it saves memory, not time. So I also wrote a **real W8A8 INT8 GEMM in Triton** ([llm/triton_int8.py](llm/triton_int8.py)) that keeps int8 all the way through — int32 accumulation on the INT8 tensor cores, a single dequant at the end via the outer product of the per-token and per-channel scales. Benchmarked against cuBLAS bf16, it **wins on large matmuls** (1.15× on the lm_head-shaped 768×16384, 1.55× on a 4096² square) but **loses on small ones** (0.34–0.90×): the crossover is exactly where the GEMM becomes compute-bound enough for INT8 tensor cores to overcome the quantization overhead — and my kernel isn't autotuned the way cuBLAS is. That turns the quantization story from "quality vs size" into "quality vs size vs latency," honestly bounded by where the win actually materializes.

## 8. A fused Triton kernel (Phase 7)

RMSNorm reads its input several times over HBM in the naive PyTorch version. The fused kernel loads each row once into SRAM, computes the norm, and writes the result — one launch per row-block, forward **and** backward (the backward fuses the input-gradient and accumulates the weight-gradient with atomics). Correctness is asserted against the PyTorch reference (fp32 and bf16, fwd and bwd). Speedup at dim 768, bf16:

| tokens | torch fwd | triton fwd | speedup |
|---|---|---|---|
| 16,384 | 1.067 ms | 0.071 ms | **15.0×** |
| 262,144 | 14.736 ms | 1.647 ms | 9.0× |

## 8b. A fused attention kernel (Phase 7, elite)

RMSNorm was the warm-up; **attention is the real fused-kernel jump.** I wrote a FlashAttention-style forward kernel in Triton ([llm/triton_attention.py](llm/triton_attention.py)): it tiles the queries, streams over key/value blocks, and keeps a running max + running sum + output accumulator in SRAM (online softmax), so it never materializes the T×T score matrix — **O(T) memory instead of O(T²)**. Causal masking is handled per key-block, with boundary masks for sequence lengths that aren't a multiple of the block size.

Correctness is pinned against `F.scaled_dot_product_attention` across 10 cases (causal / non-causal, T not a multiple of the block, and GQA via KV expansion), and it's wired into the model as an optional inference path (`model.use_triton_attention()`) that matches the SDPA path end-to-end to 8e-4.

Benchmark (B=4, H=12, D=64, fp16, causal), time in ms and peak memory in MB:

| seq len | naive | **triton (mine)** | sdpa | naive mem | **triton mem** | sdpa mem |
|---|---|---|---|---|---|---|
| 1024 | 2.86 ms | **0.30 ms** | 0.30 ms | 236 MB | **34 MB** | 34 MB |
| 2048 | 9.34 ms | **1.15 ms** | 1.09 ms | 868 MB | **59 MB** | 59 MB |
| 4096 | 46.53 ms | **4.04 ms** | 4.19 ms | 3347 MB | **109 MB** | 109 MB |

At 4k tokens my kernel is **11.5× faster and 30× lighter than naive**, and it **matches PyTorch's SDPA** — the same production FlashAttention kernel — on both axes. That's the honest headline: I didn't beat a strawman, I hit the ceiling with a hand-written kernel.

Honest scope: this is **forward-only** (inference); a fused backward (to train on it) is the next step. SDPA is itself flash, so it's the ceiling — the point is a correct hand-written fused attention and the memory/latency collapse vs naive. Reproduce: `python scripts/bench_attention.py`.

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

### A downstream benchmark: Story-Cloze

Perplexity measures fluency, not understanding. HellaSwag is far too hard for a TinyStories-scale model, so I built a domain-matched benchmark ([llm/benchmark_cloze.py](llm/benchmark_cloze.py)): strip a story's last sentence, and check whether the model assigns higher average log-prob to the **true** ending than to a **distractor** ending lifted from a different story. The distractor is a real, fluent sentence — wrong only *in context* — so this tests narrative coherence, not surface fluency. Built deterministically from held-out data; chance = 50%.

| model | Story-Cloze accuracy (500 items) |
|---|---|
| 30M | 0.956 |
| flagship 113M | 0.930 |

Both are far above chance — the models genuinely prefer coherent continuations. (The scores aren't cross-comparable to the decimal because the two models use different tokenizers, which changes the per-token averaging; the headline is "both ~0.93–0.96", not "30M > 113M".)

## 10b. Instruction tuning (Phase 5)

SFT turns "generates text" into "follows instructions." I fine-tuned the 30M base on 30k TinyStories-Instruct `(prompt, response)` pairs, with the loss **masked over the prompt** — only the response tokens are supervised (right-padded batches; causal attention means real tokens never see the trailing pad, so no mask surgery). 3 epochs, lr 2e-5, SFT val loss 1.14.

Prompt format: `Features:` / `Words:` (the story must use them) / `Summary:` (the plot to follow) / `Story:`.

- **Before** (base model): treats the fields as noise — rambles, ignores the required words, leaks literal `Story:`/`Apparent:` tokens.
- **After** (SFT): reliably incorporates every required word and matches the summary. Given *dog, jump, happy* it writes a story about a happy dog that jumps; given *boat, river, brave* it writes a brave girl sailing a boat down a river.

Full before/after in `samples/sft_before_after.md`.

### DPO — preference alignment (stretch)

I also implemented **Direct Preference Optimization from scratch** ([llm/dpo.py](llm/dpo.py)): the DPO loss with a frozen reference model (the SFT model), sequence log-probs over response tokens, no reward model or RL. Preference pairs use the real story as *chosen* and a base-model generation as *rejected*. Starting from the SFT checkpoint, held-out **preference accuracy rose 0.70 → 0.97** (fraction of pairs where the model scores the gold story above the base ramble).

Honest caveat: the implicit-reward margins over-optimized (log-prob diffs in the hundreds, train accuracy → 1.0 fast). Outputs stayed coherent because SFT anchored them, but I'd temper a real run with a higher β (0.3–0.5) and early stopping. Details in `samples/dpo_results.md`.

## 11. What didn't work / what I'd do differently

- **`torch.compile` on Windows** silently needs Triton; the first run failed with a cryptic `TritonMissing`. I made compile degrade gracefully to eager and documented `triton-windows`. Lesson: probe compilation *before* the training loop so it fails in 5 s, not mid-run.
- **Quantization speed**: I built the accuracy/size story honestly but a dequant→matmul path is not faster. Doing it "for real" means an INT8 GEMM (CUTLASS/Triton) — scoped as future work rather than faked.
- **Real-language pretraining** (FineWeb-Edu) is deliberately out of scope: trillions of tokens don't fit on a laptop and chasing it would have burned the whole budget for a worse story than a finished TinyStories model.
- If I restarted: wire W&B from day 1 (I logged to JSONL and added W&B as opt-in), and build the eval harness *before* the flagship run so every checkpoint is scored automatically.

## 12. Reproducing

Everything is in the README quickstart. Configs are versioned per experiment, checkpoints carry their config, and the results tables regenerate from the `scripts/bench_*.py` and `scripts/*_study.py` drivers. Clone, read, and — if you want — run.
