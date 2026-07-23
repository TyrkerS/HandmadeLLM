"""Correctness of the fused Triton RMSNorm vs the PyTorch reference.

Skipped automatically when Triton or CUDA is unavailable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from llm.model import RMSNorm
from llm.triton_rmsnorm import HAS_TRITON, FusedRMSNorm, fused_rmsnorm

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires Triton + CUDA",
)


@pytest.mark.parametrize("shape", [(4, 512), (2, 128, 768), (1, 1, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_matches_reference(shape, dtype):
    torch.manual_seed(0)
    dim = shape[-1]
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(dim, device="cuda", dtype=dtype)

    ref = RMSNorm(dim).cuda()
    ref.weight.data = w.clone()
    expected = ref(x)
    got = fused_rmsnorm(x, w, ref.eps)

    atol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(got, expected, atol=atol, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_matches_reference(dtype):
    torch.manual_seed(1)
    dim = 768
    x = torch.randn(8, 32, dim, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(dim, device="cuda", dtype=dtype, requires_grad=True)

    ref = RMSNorm(dim).cuda()
    ref.weight.data = w.detach().clone()
    ref.weight.requires_grad_(True)

    xr = x.detach().clone().requires_grad_(True)
    ref(xr).square().sum().backward()

    fused_rmsnorm(x, w, ref.eps).square().sum().backward()

    atol = 1e-3 if dtype == torch.float32 else 5e-2
    assert torch.allclose(x.grad, xr.grad, atol=atol, rtol=1e-2)
    assert torch.allclose(w.grad, ref.weight.grad, atol=atol, rtol=1e-2)


def test_module_trains():
    """A tiny optimization loop with FusedRMSNorm reduces a simple loss."""
    torch.manual_seed(2)
    norm = FusedRMSNorm(256).cuda()
    x = torch.randn(16, 256, device="cuda")
    target = torch.randn(16, 256, device="cuda")
    opt = torch.optim.SGD(norm.parameters(), lr=0.1)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = (norm(x) - target).pow(2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first
