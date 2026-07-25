import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

from llm.triton_int8 import HAS_TRITON, Int8Linear, int8_matmul

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires Triton + CUDA",
)


@pytest.mark.parametrize("M,K,N", [(64, 128, 64), (200, 768, 512), (512, 768, 2048)])
def test_int8_matmul_close_to_fp(M, K, N):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda")
    w = torch.randn(N, K, device="cuda")

    # quantize weights per row the same way Int8Linear does
    from llm.triton_int8 import _quant_per_row
    wq, ws = _quant_per_row(w)

    got = int8_matmul(x, wq, ws)
    exp = x @ w.T
    rel = (got - exp).norm() / exp.norm()
    assert rel < 0.05, f"relative error {rel:.4f}"


def test_int8_linear_matches_linear():
    torch.manual_seed(1)
    lin = nn.Linear(768, 512, bias=False).cuda()
    il = Int8Linear.from_linear(lin)
    x = torch.randn(128, 768, device="cuda")
    rel = (il(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.05


def test_int8_linear_batched_input():
    """Works on (B, T, K) inputs, not just (M, K)."""
    torch.manual_seed(2)
    lin = nn.Linear(256, 128, bias=False).cuda()
    il = Int8Linear.from_linear(lin)
    x = torch.randn(4, 32, 256, device="cuda")
    out = il(x)
    assert out.shape == (4, 32, 128)
    rel = (out - lin(x)).norm() / lin(x).norm()
    assert rel < 0.05
