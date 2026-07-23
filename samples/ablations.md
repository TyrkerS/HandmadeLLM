# Architecture ablations (TinyStories, fixed budget)

Single-change-at-a-time vs the modern baseline (RoPE + SwiGLU + RMSNorm + GQA).

| variant | change | final val loss | Δ vs baseline |
|---|---|---|---|
| baseline_rope_swiglu_rmsnorm_gqa | — (baseline) | 1.813 | — |
| learned_pos | RoPE → learned pos-emb | 1.891 | +0.078 |
| gelu_mlp | SwiGLU → GELU MLP | 1.860 | +0.047 |
| layernorm | RMSNorm → LayerNorm | 1.810 | -0.003 |
| mha_no_gqa | GQA → full MHA | 1.798 | -0.014 |
