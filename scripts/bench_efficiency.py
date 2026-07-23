"""Phase 2: measure what each efficiency trick buys (tokens/s, peak VRAM).

Runs short training bursts of the flagship config on random tokens (the GPU
doesn't care) with optimizations toggled cumulatively, and prints a markdown
table for the README.

Usage:
    python scripts/bench_efficiency.py                # flagship 113M
    python scripts/bench_efficiency.py --config configs/tinystories_30m.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import Config  # noqa: E402
from llm.model import Transformer  # noqa: E402


def bench(mcfg, *, dtype, grad_ckpt, compile_model, batch_size, accum, steps=8, warmup=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)

    model = Transformer(mcfg).cuda()
    model.grad_checkpointing = grad_ckpt
    if compile_model:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    ctx = torch.autocast("cuda", dtype) if dtype != torch.float32 else nullcontext()

    x = torch.randint(0, mcfg.vocab_size, (batch_size, mcfg.max_seq_len), device="cuda")
    y = torch.randint(0, mcfg.vocab_size, (batch_size, mcfg.max_seq_len), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            with ctx:
                _, loss = model(x, y)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    tok_s = batch_size * accum * mcfg.max_seq_len * steps / dt
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return tok_s, vram


def try_bench(name, mcfg, **kw):
    try:
        tok_s, vram = bench(mcfg, **kw)
        print(f"| {name} | {kw['batch_size']}x{kw['accum']} | {tok_s:,.0f} | {vram:.2f} |")
        return (name, tok_s, vram)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"| {name} | {kw['batch_size']}x{kw['accum']} | OOM | >12 |")
        return (name, None, None)
    except Exception as e:  # e.g. torch.compile unavailable on this platform
        torch.cuda.empty_cache()
        print(f"| {name} | {kw['batch_size']}x{kw['accum']} | failed: {type(e).__name__} | - |")
        return (name, None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/flagship_113m.yaml")
    args = ap.parse_args()

    mcfg = Config.from_yaml(args.config).model
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)} | config: {args.config}")
    print(f"model dim={mcfg.dim} layers={mcfg.n_layers} ctx={mcfg.max_seq_len}\n")
    print("| setup | micro-batch x accum | tokens/s | peak VRAM (GB) |")
    print("|---|---|---|---|")

    f32, bf16 = torch.float32, torch.bfloat16
    try_bench("fp32 baseline", mcfg, dtype=f32, grad_ckpt=False, compile_model=False, batch_size=8, accum=4)
    try_bench("+ bf16 autocast", mcfg, dtype=bf16, grad_ckpt=False, compile_model=False, batch_size=8, accum=4)
    try_bench("+ grad checkpointing", mcfg, dtype=bf16, grad_ckpt=True, compile_model=False, batch_size=8, accum=4)
    try_bench("+ bigger micro-batch (ckpt)", mcfg, dtype=bf16, grad_ckpt=True, compile_model=False, batch_size=24, accum=4)
    try_bench("+ torch.compile", mcfg, dtype=bf16, grad_ckpt=True, compile_model=True, batch_size=24, accum=4)


if __name__ == "__main__":
    main()
