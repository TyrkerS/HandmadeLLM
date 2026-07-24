"""Phase 7 (elite): benchmark the fused Triton attention vs naive vs SDPA.

Three implementations, increasing sequence length, causal:
  - naive : materialize the full T×T scores (the O(T²)-memory baseline fusion fixes)
  - triton: our fused FlashAttention-style kernel (O(T) memory)
  - sdpa  : F.scaled_dot_product_attention (PyTorch's own flash kernel = the ceiling)

Reports time (ms) and peak memory (MB). The honest story is memory: the fused
kernels stay flat while naive blows up; SDPA is the speed ceiling, not a rival
to beat.

Usage:
    python scripts/bench_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.triton_attention import HAS_TRITON, flash_attention  # noqa: E402


def naive_attention(q, k, v, causal=True):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        T = q.shape[-2]
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


def _time_mem(fn, iters=30, warmup=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    mem = torch.cuda.max_memory_allocated() / 1e6
    return ms, mem


def main() -> None:
    assert HAS_TRITON and torch.cuda.is_available(), "needs Triton + CUDA"
    B, H, D = 4, 12, 64
    dtype = torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | B={B} H={H} D={D} dtype={dtype}\n")
    print("| seq len | naive ms | triton ms | sdpa ms | naive MB | triton MB | sdpa MB |")
    print("|---|---|---|---|---|---|---|")

    for T in (256, 512, 1024, 2048, 4096):
        q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, T, D, device="cuda", dtype=dtype)

        try:
            n_ms, n_mb = _time_mem(lambda: naive_attention(q, k, v, True))
            n_ms, n_mb = f"{n_ms:.2f}", f"{n_mb:.0f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_ms, n_mb = "OOM", "OOM"

        t_ms, t_mb = _time_mem(lambda: flash_attention(q, k, v, True))
        s_ms, s_mb = _time_mem(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))

        print(f"| {T} | {n_ms} | {t_ms:.2f} | {s_ms:.2f} | {n_mb} | {t_mb:.0f} | {s_mb:.0f} |")


if __name__ == "__main__":
    main()
