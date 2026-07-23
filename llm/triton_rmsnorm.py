"""Phase 7: fused RMSNorm in Triton (forward + backward).

RMSNorm(x)_i = x_i / sqrt(mean(x^2) + eps) * w_i

The PyTorch reference reads x several times (square, mean, rsqrt, mul, cast).
The fused kernel loads each row once into SRAM, computes the norm, and writes
the result — one pass, one kernel launch per row-block. The backward kernel
fuses the input-gradient and accumulates the weight-gradient with atomics.

`FusedRMSNorm` is a drop-in `nn.Module` matching `llm.model.RMSNorm` numerics
(float32 reduction, cast back to input dtype). Correctness is asserted against
the reference in tests/test_triton.py; benchmark in scripts/bench_triton.py.

Falls back with a clear error if Triton isn't installed — import guarded so the
rest of the project runs without it.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd(
        X, W, Y, Rrms,
        stride_row,
        N: tl.constexpr,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        x_ptr = X + row * stride_row
        y_ptr = Y + row * stride_row
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rrms = 1.0 / tl.sqrt(mean_sq + eps)
        tl.store(Rrms + row, rrms)  # cached for backward

        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_bwd(
        X, W, DY, Rrms, DX, DW,
        stride_row,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.load(Rrms + row)

        xhat = x * rrms
        wdy = w * dy
        # dx = rrms * (wdy - xhat * mean(xhat * wdy))
        c = tl.sum(xhat * wdy, axis=0) / N
        dx = rrms * (wdy - xhat * c)
        tl.store(DX + row * stride_row + cols, dx.to(DX.dtype.element_ty), mask=mask)

        dw = dy * xhat
        tl.atomic_add(DW + cols, dw, mask=mask)

    class _FusedRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            orig_shape = x.shape
            x2d = x.reshape(-1, orig_shape[-1]).contiguous()
            M, N = x2d.shape
            y = torch.empty_like(x2d)
            rrms = torch.empty(M, device=x.device, dtype=torch.float32)
            BLOCK = triton.next_power_of_2(N)
            _rmsnorm_fwd[(M,)](x2d, weight, y, rrms, x2d.stride(0), N, eps, BLOCK=BLOCK)
            ctx.save_for_backward(x2d, weight, rrms)
            ctx.eps = eps
            ctx.orig_shape = orig_shape
            return y.reshape(orig_shape)

        @staticmethod
        def backward(ctx, dy):
            x2d, weight, rrms = ctx.saved_tensors
            M, N = x2d.shape
            dy2d = dy.reshape(-1, N).contiguous()
            dx = torch.empty_like(x2d)
            dw = torch.zeros(N, device=x2d.device, dtype=torch.float32)
            BLOCK = triton.next_power_of_2(N)
            _rmsnorm_bwd[(M,)](x2d, weight, dy2d, rrms, dx, dw, x2d.stride(0), N, BLOCK=BLOCK)
            return dx.reshape(ctx.orig_shape), dw.to(weight.dtype), None

    def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return _FusedRMSNormFn.apply(x, weight, eps)

else:  # pragma: no cover

    def fused_rmsnorm(x, weight, eps=1e-5):
        raise RuntimeError("Triton is not installed; cannot use fused_rmsnorm.")


class FusedRMSNorm(torch.nn.Module):
    """Drop-in replacement for llm.model.RMSNorm backed by the Triton kernel."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm(x, self.weight, self.eps)
