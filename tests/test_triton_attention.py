import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn.functional as F

from llm.triton_attention import HAS_TRITON, flash_attention

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires Triton + CUDA",
)


def _ref(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


@pytest.mark.parametrize("T", [16, 64, 100, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_matches_sdpa(T, causal):
    torch.manual_seed(0)
    B, H, D = 2, 4, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

    got = flash_attention(q, k, v, causal=causal)
    exp = _ref(q, k, v, causal)
    assert torch.allclose(got, exp, atol=2e-3, rtol=1e-3)


def test_non_multiple_of_block():
    """T not a multiple of BLOCK_M (64) exercises the boundary masks."""
    torch.manual_seed(1)
    q = torch.randn(1, 2, 130, 64, device="cuda")
    k = torch.randn(1, 2, 130, 64, device="cuda")
    v = torch.randn(1, 2, 130, 64, device="cuda")
    got = flash_attention(q, k, v, causal=True)
    exp = _ref(q, k, v, True)
    assert torch.allclose(got, exp, atol=2e-3, rtol=1e-3)


def test_gqa_via_expansion():
    """With KV expanded to H heads, matches SDPA on the expanded tensors."""
    torch.manual_seed(2)
    B, H, T, D, KV = 2, 6, 64, 64, 2
    q = torch.randn(B, H, T, D, device="cuda")
    k = torch.randn(B, KV, T, D, device="cuda")
    v = torch.randn(B, KV, T, D, device="cuda")
    rep = H // KV
    ke = k.repeat_interleave(rep, dim=1)
    ve = v.repeat_interleave(rep, dim=1)
    got = flash_attention(q, ke, ve, causal=True)
    exp = _ref(q, ke, ve, True)
    assert torch.allclose(got, exp, atol=2e-3, rtol=1e-3)
