"""Phase 4 (elite): a real INT8 GEMM in Triton (W8A8), so quantization actually
speeds up compute instead of only saving memory.

The Phase-4 QuantLinear dequantizes weights to bf16 and runs a normal matmul —
smaller, but not faster. This kernel keeps everything in int8 through the matmul:
weights are quantized per output channel, activations per token (dynamic), the
inner product accumulates in int32 on the INT8 tensor cores, and the result is
dequantized once at the end with the outer product of the two scale vectors.

    y[m,n] = (sum_k a_int8[m,k] * w_int8[n,k]) * a_scale[m] * w_scale[n]

`Int8Linear` is a drop-in for a bias-free nn.Linear. Correctness (vs fp matmul)
and speed are checked in tests / scripts. Requires Triton + int8 tensor cores.
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
    def _int8_gemm(
        A, W, Y, a_scale, w_scale,
        M, N, K,
        stride_am, stride_ak,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # A is (M, K) int8; W is (N, K) int8 (row = output channel), so y = A @ W^T
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0)
            w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0)
            # a: (BM, BK) int8, w: (BN, BK) int8 -> want (BM, BN): a @ w^T
            acc += tl.dot(a, tl.trans(w), out_dtype=tl.int32)
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk

        a_s = tl.load(a_scale + offs_m, mask=offs_m < M, other=0.0)
        w_s = tl.load(w_scale + offs_n, mask=offs_n < N, other=0.0)
        y = acc.to(tl.float32) * a_s[:, None] * w_s[None, :]

        y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def _quant_per_row(x: torch.Tensor):
        """Symmetric int8 per-row quantization. x: (M, K) -> (int8, scale (M,))."""
        s = x.abs().amax(dim=1, keepdim=True) / 127.0
        s = s.clamp(min=1e-8)
        q = torch.round(x / s).clamp(-127, 127).to(torch.int8)
        return q, s.squeeze(1)

    def int8_matmul(x: torch.Tensor, w_int8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
        """y = x @ w_int8.T dequantized. x: (..., K) float; w_int8: (N, K) int8;
        w_scale: (N,). Activations are dynamically quantized per row (per token)."""
        orig_shape = x.shape[:-1]
        K = x.shape[-1]
        x2d = x.reshape(-1, K).contiguous()
        a_int8, a_scale = _quant_per_row(x2d.float())
        M, N = x2d.shape[0], w_int8.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _int8_gemm[grid](
            a_int8, w_int8, y, a_scale.float(), w_scale.float(),
            M, N, K,
            a_int8.stride(0), a_int8.stride(1),
            w_int8.stride(0), w_int8.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return y.reshape(*orig_shape, N).to(x.dtype)

else:  # pragma: no cover

    def int8_matmul(x, w_int8, w_scale):
        raise RuntimeError("Triton is not installed; cannot use int8_matmul.")


class Int8Linear(torch.nn.Module):
    """Drop-in for a bias-free nn.Linear using the Triton W8A8 GEMM."""

    def __init__(self, w_int8: torch.Tensor, w_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("w_int8", w_int8)
        self.register_buffer("w_scale", w_scale)
        self.out_features, self.in_features = w_int8.shape

    @classmethod
    def from_linear(cls, lin: torch.nn.Linear) -> "Int8Linear":
        from .triton_int8 import _quant_per_row
        q, s = _quant_per_row(lin.weight.data.float())
        return cls(q, s.to(lin.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_matmul(x, self.w_int8, self.w_scale)
