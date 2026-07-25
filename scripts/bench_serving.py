"""Phase 4 (elite): continuous-batching throughput vs sequential serving.

Shows the core serving win: decoding N requests one-at-a-time leaves the GPU
idle between tokens; batching them advances all N per forward pass, so aggregate
throughput scales far past 1× until the GPU saturates.

Compares, for increasing concurrency N:
  - sequential : N requests served back-to-back (baseline, what a naive server does)
  - batched    : the ContinuousBatchingEngine serving all N together

Reports aggregate tokens/s and speedup. Uses a real checkpoint if given, else a
random-weight flagship-shaped model (throughput is weight-independent).

Usage:
    python scripts/bench_serving.py --ckpt checkpoints/tinystories_30m/best.pt
    python scripts/bench_serving.py --config configs/flagship_113m.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import Config, ModelConfig, _build  # noqa: E402
from llm.engine import ContinuousBatchingEngine, Request  # noqa: E402
from llm.model import Transformer  # noqa: E402


def load(args, device):
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = Transformer(_build(ModelConfig, ck["config"]["model"])).to(device)
        model.load_state_dict(ck["model"])
    else:
        model = Transformer(Config.from_yaml(args.config).model).to(device)
    return model.eval()


@torch.inference_mode()
def sequential(model, prompts, n, device):
    total = 0
    torch.cuda.synchronize()
    t0 = time.time()
    for p in prompts:
        idx = torch.tensor([p], device=device)
        out = model.generate(idx, max_new_tokens=n, temperature=0.8, top_k=50)
        total += out.shape[1] - len(p)
    torch.cuda.synchronize()
    return total / (time.time() - t0)


def batched(model, prompts, n, device):
    engine = ContinuousBatchingEngine(model, device=device, max_batch=len(prompts))
    reqs = [Request(id=i, tokens=list(p), prompt_len=len(p), max_new_tokens=n,
                    temperature=0.8, top_k=50) for i, p in enumerate(prompts)]
    torch.cuda.synchronize()
    t0 = time.time()
    engine.generate(reqs)
    torch.cuda.synchronize()
    total = sum(len(r.generated) for r in reqs)
    return total / (time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="configs/tinystories_30m.yaml")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args, device)
    vocab = model.cfg.vocab_size
    print(f"device: {device} | params: {model.num_params():,} | new_tokens: {args.new_tokens}\n")

    def make_prompts(n):
        g = torch.Generator().manual_seed(0)
        return [torch.randint(1, vocab, (args.prompt_len,), generator=g).tolist() for _ in range(n)]

    # warmup
    batched(model, make_prompts(2), 8, device)

    print("| concurrency | sequential tok/s | batched tok/s | speedup |")
    print("|---|---|---|---|")
    for N in (1, 2, 4, 8, 16):
        prompts = make_prompts(N)
        seq = sequential(model, [list(p) for p in prompts], args.new_tokens, device)
        bat = batched(model, [list(p) for p in prompts], args.new_tokens, device)
        print(f"| {N} | {seq:,.0f} | {bat:,.0f} | {bat/seq:.2f}x |")


if __name__ == "__main__":
    main()
