"""Phase 7 (elite stretch): fused causal attention in Triton (FlashAttention-style).

The naive attention path materializes the full T×T scores matrix in HBM. This
kernel never does: it tiles queries into blocks, streams over key/value blocks,
and keeps a running max + running sum + output accumulator in SRAM (the online-
softmax trick). Memory is O(T) per query block instead of O(T²).

Scope (honest): this is a **forward-only inference** kernel, validated against
`F.scaled_dot_product_attention`. GQA is handled by expanding KV heads to query
heads before the launch (same as the eager model). A fused backward is the
natural next step; training still uses SDPA.

Head dim must be a power of two (ours is 64). Falls back to a clear error if
Triton is missing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flash_fwd(
        Q, K, V, O,
        scale,
        stride_qz, stride_qt, stride_qd,
        stride_kz, stride_kt, stride_kd,
        stride_vz, stride_vt, stride_vd,
        stride_oz, stride_ot, stride_od,
        T,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        pid_m = tl.program_id(0)       # query block index
        zh = tl.program_id(1)          # flattened (batch * head) index

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptr = Q + zh * stride_qz + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptr, mask=offs_m[:, None] < T, other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        # causal: keys only up to the last query row in this block
        n_end = (pid_m + 1) * BLOCK_M if CAUSAL else T

        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptr = K + zh * stride_kz + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k = tl.load(k_ptr, mask=offs_n[:, None] < T, other=0.0).to(tl.float32)

            qk = tl.dot(q, tl.trans(k)) * scale        # (BLOCK_M, BLOCK_N)
            mask = offs_n[None, :] < T
            if CAUSAL:
                mask = mask & (offs_m[:, None] >= offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)  # rows with no valid key
            alpha = tl.exp(m_i - m_ij)
            alpha = tl.where(m_ij == float("-inf"), 1.0, alpha)

            v_ptr = V + zh * stride_vz + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptr, mask=offs_n[:, None] < T, other=0.0).to(tl.float32)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        o_ptr = O + zh * stride_oz + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        tl.store(o_ptr, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < T)

    def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        causal: bool = True) -> torch.Tensor:
        """q,k,v: (B, H, T, HEAD_DIM). Returns (B, H, T, HEAD_DIM). Forward only.

        GQA: pass k,v already expanded to H heads (repeat_interleave), matching q.
        """
        B, H, T, D = q.shape
        assert k.shape == q.shape and v.shape == q.shape, "expand KV to H heads first"
        assert (D & (D - 1)) == 0, "HEAD_DIM must be a power of two"
        scale = 1.0 / (D ** 0.5)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        qf = q.view(B * H, T, D)
        kf = k.view(B * H, T, D)
        vf = v.view(B * H, T, D)
        o = torch.empty_like(qf)

        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(T, BLOCK_M), B * H)
        _flash_fwd[grid](
            qf, kf, vf, o, scale,
            qf.stride(0), qf.stride(1), qf.stride(2),
            kf.stride(0), kf.stride(1), kf.stride(2),
            vf.stride(0), vf.stride(1), vf.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            T, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
        )
        return o.view(B, H, T, D)

else:  # pragma: no cover

    def flash_attention(q, k, v, causal=True):
        raise RuntimeError("Triton is not installed; cannot use flash_attention.")
