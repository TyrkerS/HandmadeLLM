import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

from llm.config import ModelConfig
from llm.model import GELUMLP, RMSNorm, SwiGLU, Transformer


def _cfg(**kw):
    base = dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=100, max_seq_len=32)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("pos_emb", ["rope", "learned"])
@pytest.mark.parametrize("mlp", ["swiglu", "gelu"])
@pytest.mark.parametrize("norm", ["rmsnorm", "layernorm"])
def test_variant_forward_backward(pos_emb, mlp, norm):
    torch.manual_seed(0)
    model = Transformer(_cfg(pos_emb=pos_emb, mlp=mlp, norm=norm))
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 16, 100)
    loss.backward()
    assert model.tok_emb.weight.grad is not None


def test_flags_select_modules():
    rope = Transformer(_cfg(pos_emb="rope"))
    assert rope.pos_emb is None
    assert rope.blocks[0].attn.use_rope

    learned = Transformer(_cfg(pos_emb="learned"))
    assert isinstance(learned.pos_emb, nn.Embedding)
    assert not learned.blocks[0].attn.use_rope

    assert isinstance(Transformer(_cfg(mlp="swiglu")).blocks[0].mlp, SwiGLU)
    assert isinstance(Transformer(_cfg(mlp="gelu")).blocks[0].mlp, GELUMLP)
    assert isinstance(Transformer(_cfg(norm="rmsnorm")).blocks[0].attn_norm, RMSNorm)
    assert isinstance(Transformer(_cfg(norm="layernorm")).blocks[0].attn_norm, nn.LayerNorm)


def test_gqa_vs_mha_param_difference():
    gqa = Transformer(_cfg(n_heads=4, n_kv_heads=2)).num_params()
    mha = Transformer(_cfg(n_heads=4, n_kv_heads=4)).num_params()
    assert mha > gqa  # MHA has more KV projection params


def test_mlp_param_counts_close():
    """SwiGLU (3 matrices) and the matched GELU MLP should be within ~10% params."""
    sw = sum(p.numel() for p in SwiGLU(_cfg()).parameters())
    ge = sum(p.numel() for p in GELUMLP(_cfg()).parameters())
    assert abs(sw - ge) / sw < 0.15
