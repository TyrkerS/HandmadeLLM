"""Phase 7: benchmark the fused Triton RMSNorm vs the PyTorch reference.

Times forward and forward+backward across sequence lengths at the flagship's
hidden dim, on the actual GPU.

Usage:
    python scripts/bench_triton.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.model import RMSNorm  # noqa: E402
from llm.triton_rmsnorm import HAS_TRITON, fused_rmsnorm  # noqa: E402


def _time(fn, iters=100, warmup=20):
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
    return start.elapsed_time(end) / iters  # ms


def main() -> None:
    assert HAS_TRITON and torch.cuda.is_available(), "needs Triton + CUDA"
    dim = 768
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)} | dim={dim} dtype={dtype}\n")

    ref = RMSNorm(dim).cuda()
    w = ref.weight.detach()

    print("| tokens (B*T) | torch fwd (ms) | triton fwd (ms) | fwd speedup | torch fwd+bwd | triton fwd+bwd | bwd speedup |")
    print("|---|---|---|---|---|---|---|")
    for n_tokens in (4096, 16384, 65536, 262144):
        x = torch.randn(n_tokens, dim, device="cuda", dtype=dtype)

        t_fwd = _time(lambda: ref(x))
        tr_fwd = _time(lambda: fused_rmsnorm(x, w, ref.eps))

        def torch_fb():
            xr = x.clone().requires_grad_(True)
            ref(xr).sum().backward()

        def triton_fb():
            xr = x.clone().requires_grad_(True)
            fused_rmsnorm(xr, w, ref.eps).sum().backward()

        t_fb = _time(torch_fb, iters=50)
        tr_fb = _time(triton_fb, iters=50)

        print(
            f"| {n_tokens:,} | {t_fwd:.3f} | {tr_fwd:.3f} | {t_fwd / tr_fwd:.2f}x "
            f"| {t_fb:.3f} | {tr_fb:.3f} | {t_fb / tr_fb:.2f}x |"
        )


if __name__ == "__main__":
    main()
