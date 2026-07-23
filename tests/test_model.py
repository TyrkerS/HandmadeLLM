import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from llm.config import ModelConfig
from llm.model import KVCache, RMSNorm, Transformer, apply_rope, precompute_rope

CFG = ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=97, max_seq_len=64)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = Transformer(CFG)
    m.eval()
    return m


def test_output_shape(model):
    x = torch.randint(0, CFG.vocab_size, (3, 16))
    logits, loss = model(x, targets=x)
    assert logits.shape == (3, 16, CFG.vocab_size)
    assert loss.item() > 0


def test_causality(model):
    """Perturbing a future token must not change logits at earlier positions."""
    torch.manual_seed(1)
    x = torch.randint(0, CFG.vocab_size, (1, 20))
    logits_a, _ = model(x, targets=x)
    x2 = x.clone()
    x2[0, 15] = (x2[0, 15] + 1) % CFG.vocab_size
    logits_b, _ = model(x2, targets=x2)
    assert torch.allclose(logits_a[0, :15], logits_b[0, :15], atol=1e-5)
    assert not torch.allclose(logits_a[0, 15:], logits_b[0, 15:], atol=1e-5)


def test_kv_cache_matches_full_forward(model):
    """Greedy generation with KV cache must equal generation without it."""
    torch.manual_seed(2)
    prompt = torch.randint(0, CFG.vocab_size, (2, 8))
    out_cache = model.generate(prompt, max_new_tokens=20, temperature=0.0, use_cache=True)
    out_full = model.generate(prompt, max_new_tokens=20, temperature=0.0, use_cache=False)
    assert torch.equal(out_cache, out_full)


def test_kv_cache_incremental_logits(model):
    """Feeding tokens one at a time through the cache = one full forward."""
    torch.manual_seed(3)
    x = torch.randint(0, CFG.vocab_size, (1, 10))
    full_logits, _ = model(x, targets=x)

    caches = [KVCache() for _ in range(CFG.n_layers)]
    with torch.inference_mode():
        for t in range(10):
            step_logits, _ = model(x[:, t : t + 1], caches=caches, start_pos=t)
    assert torch.allclose(step_logits[0, -1], full_logits[0, -1], atol=1e-4)


def test_gqa_kv_projection_shapes(model):
    attn = model.blocks[0].attn
    assert attn.wk.out_features == CFG.n_kv_heads * (CFG.dim // CFG.n_heads)
    assert attn.wq.out_features == CFG.dim
    assert attn.n_rep == CFG.n_heads // CFG.n_kv_heads


def test_mha_special_case():
    """n_kv_heads == n_heads degenerates to standard MHA and still runs."""
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=4, n_kv_heads=4, vocab_size=50, max_seq_len=32)
    m = Transformer(cfg).eval()
    x = torch.randint(0, 50, (2, 12))
    logits, _ = m(x, targets=x)
    assert logits.shape == (2, 12, 50)


def test_weight_tying(model):
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_rope_preserves_norm():
    cos, sin = precompute_rope(16, 32, 10000.0)
    q = torch.randn(1, 2, 32, 16)
    k = torch.randn(1, 2, 32, 16)
    q2, k2 = apply_rope(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-5)


def test_rope_relative_positions():
    """RoPE attention scores depend only on relative distance between q and k."""
    cos, sin = precompute_rope(16, 64, 10000.0)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(qpos, kpos):
        qr, _ = apply_rope(q, q, cos[qpos : qpos + 1], sin[qpos : qpos + 1])
        kr, _ = apply_rope(k, k, cos[kpos : kpos + 1], sin[kpos : kpos + 1])
        return (qr * kr).sum().item()

    assert abs(score(5, 3) - score(25, 23)) < 1e-4  # same distance, same score
    assert abs(score(5, 3) - score(5, 4)) > 1e-6    # different distance differs


def test_rmsnorm():
    norm = RMSNorm(32)
    x = torch.randn(4, 8, 32) * 10
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_overfit_tiny_cpu():
    """Loss decreases sharply when overfitting a fixed batch (CPU, fast)."""
    torch.manual_seed(0)
    cfg = ModelConfig(dim=32, n_layers=2, n_heads=2, n_kv_heads=1, vocab_size=50, max_seq_len=16)
    m = Transformer(cfg)
    x = torch.randint(0, 50, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    _, first = m(x, targets=x)
    for _ in range(60):
        opt.zero_grad()
        _, loss = m(x, targets=x)
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.3
