"""Phase 4 (elite): does the Triton INT8 GEMM actually run faster than bf16?

Times y = x @ Wᵀ at transformer-shaped sizes with:
  - bf16 : torch.nn.functional.linear in bf16 (cuBLAS, the baseline to beat)
  - int8 : our Triton W8A8 kernel (dynamic per-token activation quant included)

Honest test: INT8 tensor cores only win once the matmul is big enough to be
compute-bound. This prints the crossover on the actual GPU.

Usage:
    python scripts/bench_int8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.triton_int8 import HAS_TRITON, _quant_per_row, int8_matmul  # noqa: E402


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    assert HAS_TRITON and torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print("| M (tokens) | K×N | bf16 ms | int8 ms | speedup |")
    print("|---|---|---|---|---|")
    # (K, N): MLP up (768->2048), lm_head (768->16384), a big square
    for (K, N) in [(768, 2048), (768, 16384), (4096, 4096)]:
        for M in (512, 2048):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda")
            wq, ws = _quant_per_row(w)
            wb = w.to(torch.bfloat16)

            t_bf16 = _time(lambda: F.linear(x, wb))
            t_int8 = _time(lambda: int8_matmul(x, wq, ws))
            print(f"| {M} | {K}×{N} | {t_bf16:.3f} | {t_int8:.3f} | {t_bf16/t_int8:.2f}x |")


if __name__ == "__main__":
    main()
