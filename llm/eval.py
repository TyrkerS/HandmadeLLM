"""Phase 6: evaluation harness — perplexity and bits-per-byte on held-out data.

Perplexity = exp(mean token-level cross-entropy) over a token stream, computed
in non-overlapping windows of the model's context length. Perplexity depends on
the tokenizer, so it is NOT comparable across models with different vocabs.

Bits-per-byte normalizes the same total negative log-likelihood by the number of
UTF-8 bytes the tokens represent, giving a tokenizer-independent metric that
*is* comparable across models (this is the fair way to rank e.g. the 8k-vocab
30M against the 16k-vocab flagship).

Usage:
    python -m llm.eval --ckpt checkpoints/tinystories_30m/best.pt --bin data/tinystories/val.bin
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .bpe import BPETokenizer
from .config import ModelConfig, _build
from .data import load_bin
from .model import Transformer


@torch.no_grad()
def eval_nll(model: Transformer, data: np.ndarray, seq_len: int, device: str,
            max_windows: int | None = None, byte_len: np.ndarray | None = None):
    """Returns (total_nll_nats, total_tokens, total_bytes) over held-out windows.

    `byte_len[i]` = number of UTF-8 bytes token id i decodes to; when provided,
    total_bytes is accumulated over the target tokens for bits-per-byte.
    """
    model.eval()
    n_windows = (len(data) - 1) // seq_len
    if max_windows:
        n_windows = min(n_windows, max_windows)
    total_nll, total_tokens, total_bytes = 0.0, 0, 0
    for w in range(n_windows):
        i = w * seq_len
        x = torch.from_numpy(data[i : i + seq_len].astype(np.int64))[None].to(device)
        y = torch.from_numpy(data[i + 1 : i + 1 + seq_len].astype(np.int64))[None].to(device)
        _, loss = model(x, y)
        total_nll += loss.item() * y.numel()
        total_tokens += y.numel()
        if byte_len is not None:
            total_bytes += int(byte_len[data[i + 1 : i + 1 + seq_len]].sum())
    return total_nll, total_tokens, total_bytes


def perplexity(model, data, seq_len, device, max_windows=None) -> float:
    nll, ntok, _ = eval_nll(model, data, seq_len, device, max_windows)
    return math.exp(nll / ntok)


def byte_lengths(tok: BPETokenizer) -> np.ndarray:
    """Per-token-id UTF-8 byte count, for bits-per-byte."""
    return np.array([len(tok.vocab[i]) for i in range(tok.vocab_size)], dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--tokenizer", default=None, help="defaults to the ckpt's tokenizer")
    ap.add_argument("--max-windows", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])

    tok = BPETokenizer.load(args.tokenizer or ckpt["config"]["train"]["tokenizer_path"])
    data = load_bin(args.bin)
    nll, ntok, nbytes = eval_nll(model, data, mcfg.max_seq_len, device,
                                 args.max_windows, byte_lengths(tok))
    ppl = math.exp(nll / ntok)
    bpb = (nll / math.log(2)) / nbytes
    print(f"{args.bin} ({args.max_windows} windows):")
    print(f"  perplexity     : {ppl:.3f}   (tokenizer-dependent, not cross-model)")
    print(f"  bits-per-byte  : {bpb:.4f}   (tokenizer-independent, cross-model comparable)")


if __name__ == "__main__":
    main()
