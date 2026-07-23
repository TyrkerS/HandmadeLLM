import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

from llm.config import ModelConfig
from llm.model import Transformer
from llm.quant import (
    QuantLinear,
    _pack_int4,
    _unpack_int4,
    linear_weight_bytes,
    quantize_model_,
    quantize_rowwise,
)


@pytest.mark.parametrize("bits", [8, 4])
def test_quant_dequant_roundtrip_bounded(bits):
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    q, scale = quantize_rowwise(w, bits)
    deq = q.float() * scale[:, None]
    # error bounded by half a quantization step per row
    step = scale[:, None]
    assert (deq - w).abs().max() <= step.max() * 0.75


def test_int4_pack_unpack():
    torch.manual_seed(1)
    q = torch.randint(-8, 8, (16, 64), dtype=torch.int8)
    packed = _pack_int4(q)
    assert packed.shape == (16, 32)
    assert torch.equal(_unpack_int4(packed, 64), q)


@pytest.mark.parametrize("bits", [8, 4])
def test_quantlinear_close_to_linear(bits):
    torch.manual_seed(2)
    lin = nn.Linear(256, 128, bias=False)
    ql = QuantLinear.from_linear(lin, bits)
    x = torch.randn(4, 256)
    out_ref = lin(x)
    out_q = ql(x)
    rel = (out_q - out_ref).norm() / out_ref.norm()
    assert rel < (0.05 if bits == 8 else 0.25)


def test_quantize_model_shrinks_and_runs():
    cfg = ModelConfig(dim=128, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=64)
    model = Transformer(cfg).eval()
    x = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        ref, _ = model(x, targets=x)

    bytes_before = linear_weight_bytes(model)
    quantize_model_(model, bits=8)
    bytes_after = linear_weight_bytes(model)

    assert bytes_after < bytes_before * 0.35  # fp32 -> int8 ~4x on quantized layers
    with torch.no_grad():
        out, _ = model(x, targets=x)
    assert out.shape == ref.shape
    # lm_head skipped (tied), so output distribution stays reasonable
    assert (out - ref).norm() / ref.norm() < 0.1


def test_lm_head_not_quantized():
    cfg = ModelConfig(dim=64, n_layers=1, n_heads=2, n_kv_heads=1, vocab_size=100, max_seq_len=32)
    model = Transformer(cfg)
    quantize_model_(model, bits=8)
    assert isinstance(model.lm_head, nn.Linear)  # untouched
    assert isinstance(model.blocks[0].attn.wq, QuantLinear)
