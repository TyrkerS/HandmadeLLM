"""Phase 4: KV-cache generation speedup benchmark.

Compares autoregressive decoding with and without the KV cache, and reports
latency percentiles. Uses a real checkpoint if given, else a random-weight
model of the flagship config (speed is weight-independent).

Usage:
    python scripts/bench_inference.py --ckpt checkpoints/tinystories_30m/best.pt
    python scripts/bench_inference.py --config configs/flagship_113m.yaml   # random weights
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import Config, ModelConfig, _build  # noqa: E402
from llm.model import Transformer  # noqa: E402


def load(args, device):
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        mcfg = _build(ModelConfig, ckpt["config"]["model"])
        model = Transformer(mcfg).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        mcfg = Config.from_yaml(args.config).model
        model = Transformer(mcfg).to(device)
    model.eval()
    return model, mcfg


@torch.inference_mode()
def time_generate(model, prompt, new_tokens, use_cache, runs):
    lat = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(prompt, max_new_tokens=new_tokens, temperature=0.0, use_cache=use_cache)
        torch.cuda.synchronize()
        lat.append(time.time() - t0)
    return np.array(lat)


def report(name, lat, new_tokens):
    tok_s = new_tokens / lat.mean()
    p50, p95 = np.percentile(lat * 1000, [50, 95])
    print(f"| {name} | {tok_s:,.0f} | {p50:.0f} | {p95:.0f} |")
    return tok_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="configs/tinystories_30m.yaml")
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--prompt-len", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg = load(args, device)
    print(f"device: {device} | params: {model.num_params():,} | new_tokens: {args.new_tokens}\n")

    prompt = torch.randint(0, mcfg.vocab_size, (1, args.prompt_len), device=device)

    # warmup (compile/cudnn autotune)
    time_generate(model, prompt, 16, True, 2)

    print("| mode | tokens/s | p50 latency (ms) | p95 latency (ms) |")
    print("|---|---|---|---|")
    lat_cache = time_generate(model, prompt, args.new_tokens, True, args.runs)
    lat_nocache = time_generate(model, prompt, args.new_tokens, False, args.runs)
    tok_cache = report("with KV cache", lat_cache, args.new_tokens)
    tok_nocache = report("without KV cache", lat_nocache, args.new_tokens)
    print(f"\nKV-cache speedup: {tok_cache / tok_nocache:.2f}x")


if __name__ == "__main__":
    main()
