"""Phase 4: quantization quality/size/speed trade-off on a trained checkpoint.

For fp (baseline), int8 and int4 weight-only quantization, reports:
  - held-out perplexity (quality)
  - total linear-weight bytes (size)
  - generation throughput (speed)

Usage:
    python scripts/bench_quant.py --ckpt checkpoints/tinystories_30m/best.pt \
        --bin data/tinystories/val.bin
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import ModelConfig, _build  # noqa: E402
from llm.data import load_bin  # noqa: E402
from llm.eval import perplexity  # noqa: E402
from llm.model import Transformer  # noqa: E402
from llm.quant import linear_weight_bytes, quantize_model_  # noqa: E402


@torch.inference_mode()
def gen_throughput(model, mcfg, device, new_tokens=128, runs=5):
    prompt = torch.randint(0, mcfg.vocab_size, (1, 16), device=device)
    model.generate(prompt, max_new_tokens=8, temperature=0.0)  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        model.generate(prompt, max_new_tokens=new_tokens, temperature=0.0)
    torch.cuda.synchronize()
    return new_tokens * runs / (time.time() - t0)


def build(ckpt, device):
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    m = Transformer(mcfg).to(device)
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, mcfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--max-windows", type=int, default=300)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    data = load_bin(args.bin)

    print(f"device: {device} | ckpt: {args.ckpt}\n")
    print("| precision | linear weights (MB) | perplexity | gen tok/s |")
    print("|---|---|---|---|")

    rows = []
    for label, bits in [("fp (baseline)", None), ("int8", 8), ("int4", 4)]:
        model, mcfg = build(ckpt, device)
        if bits is not None:
            quantize_model_(model, bits=bits)
        mb = linear_weight_bytes(model) / 1e6
        ppl = perplexity(model, data, mcfg.max_seq_len, device, args.max_windows)
        tok_s = gen_throughput(model, mcfg, device) if device == "cuda" else float("nan")
        print(f"| {label} | {mb:.1f} | {ppl:.2f} | {tok_s:,.0f} |")
        rows.append((label, mb, ppl))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    base_mb, base_ppl = rows[0][1], rows[0][2]
    print("\nvs fp baseline:")
    for label, mb, ppl in rows[1:]:
        print(f"  {label}: {base_mb / mb:.1f}x smaller, perplexity {ppl / base_ppl - 1:+.1%}")


if __name__ == "__main__":
    main()
