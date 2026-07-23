"""Phase 6: evaluation harness — perplexity on held-out data.

Perplexity = exp(mean token-level cross-entropy) over a token stream, computed
in non-overlapping windows of the model's context length.

Usage:
    python -m llm.eval --ckpt checkpoints/tinystories_30m/best.pt --bin data/tinystories/val.bin
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .config import ModelConfig, _build
from .data import load_bin
from .model import Transformer


@torch.no_grad()
def perplexity(model: Transformer, data: np.ndarray, seq_len: int, device: str, max_windows: int | None = None) -> float:
    model.eval()
    n_windows = (len(data) - 1) // seq_len
    if max_windows:
        n_windows = min(n_windows, max_windows)
    total_nll, total_tokens = 0.0, 0
    for w in range(n_windows):
        i = w * seq_len
        x = torch.from_numpy(data[i : i + seq_len].astype(np.int64))[None].to(device)
        y = torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64))[None].to(device)
        _, loss = model(x, y)
        total_nll += loss.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_nll / total_tokens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--max-windows", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])

    data = load_bin(args.bin)
    ppl = perplexity(model, data, mcfg.max_seq_len, device, args.max_windows)
    print(f"perplexity ({args.bin}, {args.max_windows} windows): {ppl:.2f}")


if __name__ == "__main__":
    main()
