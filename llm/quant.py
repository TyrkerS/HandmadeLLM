"""Phase 4: weight-only quantization from scratch (int8 and int4).

Linear weights dominate the parameter count, so we quantize `nn.Linear.weight`
to low-bit integers with a per-output-channel (per-row) symmetric scale, and
dequantize on the fly in the forward pass. Activations stay in bf16/fp32.

- int8: one int8 per weight, per-row fp scale. ~4x smaller than fp32 weights.
- int4: two weights packed per byte, per-row scale. ~8x smaller.

This is the honest, framework-free version used to measure the quality vs size
vs speed trade-off (scripts/bench_quant.py). It is not a fast INT8 GEMM — the
point is to show the accuracy impact and memory savings of the scheme, with
dequant-then-matmul in bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_rowwise(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row quantization. Returns (int codes, fp scale per row)."""
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(w / scale).clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scale.squeeze(1)


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack int4 values (stored in int8, range [-8,7]) two-per-byte."""
    out_features, in_features = q.shape
    assert in_features % 2 == 0, "int4 packing needs even in_features"
    u = (q + 8).to(torch.uint8)  # [0,15]
    return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    out = torch.empty((packed.shape[0], in_features), dtype=torch.int8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


class QuantLinear(nn.Module):
    """Drop-in for nn.Linear (bias-free) with int8/int4 weights, dequant on forward."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, in_features: int, bits: int):
        super().__init__()
        self.bits = bits
        self.in_features = in_features
        self.out_features = q.shape[0]
        if bits == 4:
            self.register_buffer("qweight", _pack_int4(q))
        else:
            self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int) -> "QuantLinear":
        q, scale = quantize_rowwise(lin.weight.data.float(), bits)
        return cls(q, scale.to(lin.weight.dtype), lin.in_features, bits)

    def _dequant_weight(self, dtype) -> torch.Tensor:
        q = _unpack_int4(self.qweight, self.in_features) if self.bits == 4 else self.qweight
        return q.to(dtype) * self.scale.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._dequant_weight(x.dtype))

    def nbytes(self) -> int:
        return self.qweight.numel() * self.qweight.element_size() + self.scale.numel() * self.scale.element_size()


def quantize_model_(model: nn.Module, bits: int, skip: tuple[str, ...] = ("lm_head",)) -> nn.Module:
    """In-place: replace bias-free nn.Linear with QuantLinear. Skips names in `skip`
    (lm_head is tied to the embedding and kept in full precision)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.bias is None and name not in skip:
            setattr(model, name, QuantLinear.from_linear(child, bits))
        else:
            quantize_model_(child, bits, skip)
    return model


def linear_weight_bytes(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.bias is None:
            total += m.weight.numel() * m.weight.element_size()
        elif isinstance(m, QuantLinear):
            total += m.nbytes()
    return total
