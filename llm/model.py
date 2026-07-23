"""Llama-style decoder-only transformer, from scratch.

Architecture: RoPE + RMSNorm (pre-norm) + causal attention with GQA + SwiGLU
MLP + weight tying. Attention uses torch's scaled_dot_product_attention so we
get flash/memory-efficient kernels for free; a custom Triton kernel is a
later-phase replacement for RMSNorm/attention.

Generation supports an optional KV cache (see `KVCache`); tests assert that
cached and uncached greedy decoding produce identical tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


# --------------------------------------------------------------------- RoPE


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each (max_seq_len, head_dim), float32."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)         # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q: (B, n_heads, T, hd), k: (B, n_kv_heads, T, hd), cos/sin: (T, hd)."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ------------------------------------------------------------------ modules


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class KVCache:
    """Per-layer key/value cache for autoregressive decoding."""

    def __init__(self):
        self.k: torch.Tensor | None = None  # (B, n_kv_heads, T_cached, hd)
        self.v: torch.Tensor | None = None

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v

    @property
    def seq_len(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        if self.n_rep > 1:  # GQA: expand KV heads to match query heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # With a warm cache we decode one token attending to all cached keys,
        # so the mask must not be causal-by-position-0; T == k_len only on
        # prefill / training, which is exactly when causal masking applies.
        is_causal = T == k.shape[2] and T > 1
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.dim / 3)
        hidden = cfg.ffn_multiple_of * math.ceil(hidden / cfg.ffn_multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)  # up
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual-stream output projections (GPT-2 style)
        res_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=res_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_weights:
            n -= self.tok_emb.weight.numel()
        elif non_embedding:
            n -= self.tok_emb.weight.numel()  # tied: subtract the shared matrix once
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
    ):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, cos, sin, cache, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache)
        x = self.norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        # inference: only the last position's logits are needed
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. temperature=0 means greedy argmax."""
        self.eval()
        caches = [KVCache() for _ in self.blocks] if use_cache else None
        pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            if idx.shape[1] >= self.cfg.max_seq_len:
                break
            if use_cache:
                logits, _ = self(cur, caches=caches, start_pos=pos)
                pos += cur.shape[1]
            else:
                logits, _ = self(idx)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            cur = next_id
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx
