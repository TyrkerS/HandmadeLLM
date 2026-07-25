"""Continuous batching inference engine (the core of what makes vLLM fast).

A naive server decodes one request at a time, leaving the GPU idle between
tokens. This engine keeps a *dynamic batch* of in-flight requests and advances
them all by one token per forward pass — and admits new requests mid-flight
(continuous batching) instead of waiting for the batch to drain.

The tricky parts, handled here:
  - **ragged lengths**: requests have different prompt/output lengths, so the
    batched KV cache is left-padded to the longest active sequence, with a
    key-padding mask so queries never attend to pad slots.
  - **per-sequence RoPE**: each row is at a different absolute position, so we
    pass per-sequence `position_ids` (not a shared `start_pos`).
  - **admission/eviction**: finished sequences are dropped and free rows are
    reused by new arrivals, keeping the batch full.

This is a single-process, synchronous scheduler — enough to demonstrate the
throughput win (see scripts/bench_serving.py). It is not multi-GPU or paged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .model import Transformer

NEG_INF = float("-inf")


@dataclass
class Request:
    id: int
    tokens: list[int]              # prompt + generated so far
    prompt_len: int
    max_new_tokens: int
    temperature: float = 0.8
    top_k: int = 50
    eos_id: int | None = None
    done: bool = False
    generated: list[int] = field(default_factory=list)


class ContinuousBatchingEngine:
    def __init__(self, model: Transformer, device: str = "cuda", max_batch: int = 16):
        self.model = model.eval()
        self.device = device
        self.max_batch = max_batch
        self.cfg = model.cfg
        self.n_layers = cfg = model.cfg.n_layers
        self.n_kv = model.cfg.n_kv_heads
        self.hd = model.cfg.dim // model.cfg.n_heads

    @torch.inference_mode()
    def _prefill(self, req: Request):
        """Run the prompt through the model, returning per-layer (k, v) caches
        of shape (1, n_kv, prompt_len, hd) and the next-token logits."""
        idx = torch.tensor([req.tokens], device=self.device)
        from .model import KVCache
        caches = [KVCache() for _ in range(self.n_layers)]
        logits, _ = self.model(idx, caches=caches, start_pos=0)
        kv = [(c.k, c.v) for c in caches]
        return kv, logits[:, -1, :]

    def _sample(self, logits: torch.Tensor, req: Request) -> int:
        if req.temperature <= 0:
            return int(logits.argmax(-1))
        logits = logits / req.temperature
        if req.top_k:
            kth = torch.topk(logits, min(req.top_k, logits.size(-1))).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, NEG_INF)
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1))

    @torch.inference_mode()
    def generate(self, requests: list[Request]) -> list[Request]:
        """Run all requests to completion with continuous batching. Returns them
        with `.generated` filled in. New requests are admitted as slots free up."""
        pending = list(requests)
        active: list[Request] = []
        # per-active-request layer caches, each (1, n_kv, L_i, hd), and lengths
        kvs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        lengths: list[int] = []

        def admit():
            while pending and len(active) < self.max_batch:
                req = pending.pop(0)
                kv, last_logits = self._prefill(req)
                tok = self._sample(last_logits, req)
                req.generated.append(tok)
                req.tokens.append(tok)
                active.append(req)
                kvs.append([(k, v) for (k, v) in kv])
                lengths.append(kv[0][0].shape[2] + 1)  # prompt + the just-sampled token
                if req.eos_id is not None and tok == req.eos_id:
                    req.done = True

        admit()
        while active:
            # decode one step for the whole active batch
            B = len(active)
            Lmax = max(lengths)
            # build left-padded batched K/V per layer + a key-padding mask
            mask = torch.zeros(B, 1, 1, Lmax, device=self.device)
            for b, L in enumerate(lengths):
                if L < Lmax:
                    mask[b, :, :, : Lmax - L] = NEG_INF  # pad slots on the left

            # last generated token per row is the query; its position = length-1
            next_tokens = torch.tensor([[r.tokens[-1]] for r in active], device=self.device)
            position_ids = torch.tensor([[L - 1] for L in lengths], device=self.device)

            x = self.model.tok_emb(next_tokens)
            if self.model.pos_emb is not None:
                x = x + self.model.pos_emb(position_ids)
            x = self.model.drop(x)

            cos = self.model.rope_cos[position_ids]
            sin = self.model.rope_sin[position_ids]

            for li, block in enumerate(self.model.blocks):
                attn = block.attn
                h = block.attn_norm(x)
                q = attn.wq(h).view(B, 1, attn.n_heads, attn.head_dim).transpose(1, 2)
                k = attn.wk(h).view(B, 1, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
                v = attn.wv(h).view(B, 1, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
                if attn.use_rope:
                    from .model import apply_rope
                    q, k = apply_rope(q, k, cos, sin)

                # append new k/v to each row's cache (right side), left-pad to Lmax
                k_pad = torch.zeros(B, attn.n_kv_heads, Lmax, attn.head_dim, device=self.device, dtype=q.dtype)
                v_pad = torch.zeros_like(k_pad)
                for b in range(B):
                    ck, cv = kvs[b][li]
                    ck = torch.cat([ck, k[b : b + 1]], dim=2)  # (1, n_kv, L, hd)
                    cv = torch.cat([cv, v[b : b + 1]], dim=2)
                    kvs[b][li] = (ck, cv)
                    L = ck.shape[2]
                    k_pad[b, :, Lmax - L :] = ck[0]
                    v_pad[b, :, Lmax - L :] = cv[0]

                kk = k_pad.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else k_pad
                vv = v_pad.repeat_interleave(attn.n_rep, dim=1) if attn.n_rep > 1 else v_pad
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
                y = y.transpose(1, 2).contiguous().view(B, 1, -1)
                x = x + attn.wo(y)
                x = x + block.mlp(block.mlp_norm(x))

            logits = self.model.lm_head(self.model.norm(x))[:, -1, :]

            # sample + book-keeping; grow each row's length
            finished_idx = []
            for b, req in enumerate(active):
                tok = self._sample(logits[b], req)
                req.generated.append(tok)
                req.tokens.append(tok)
                lengths[b] += 1
                stop = (req.eos_id is not None and tok == req.eos_id) or \
                       len(req.generated) >= req.max_new_tokens or \
                       lengths[b] >= self.cfg.max_seq_len
                if stop:
                    req.done = True
                    finished_idx.append(b)

            # evict finished (reverse order to keep indices valid), then admit new
            for b in sorted(finished_idx, reverse=True):
                active.pop(b)
                kvs.pop(b)
                lengths.pop(b)
            admit()

        return requests
