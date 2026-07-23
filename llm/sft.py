"""Phase 5: supervised fine-tuning (instruction tuning) with prompt-masked loss.

Loads a pretrained checkpoint and fine-tunes it on (prompt, response) pairs.
Cross-entropy is computed only over response tokens — the prompt is context,
not a target. Batches are right-padded; causal attention means real tokens
never attend to trailing pad, so no attention-mask surgery is needed (pad
positions are simply excluded from the loss).

Usage:
    python -m llm.sft --init checkpoints/tinystories_30m/best.pt \
        --data data/sft/train.pt --val data/sft/val.pt --out checkpoints/sft_30m
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, _build
from .model import Transformer

IGNORE = -100


def make_batch(examples, device):
    """Right-pad a list of {ids, prompt_len} into (x, targets) with prompt/pad masked."""
    maxlen = max(len(e["ids"]) for e in examples)
    B = len(examples)
    x = torch.zeros(B, maxlen, dtype=torch.long)
    tgt = torch.full((B, maxlen), IGNORE, dtype=torch.long)
    for i, e in enumerate(examples):
        ids = e["ids"]
        n = len(ids)
        x[i, :n] = torch.tensor(ids)
        # targets are next-token; supervise only response positions
        # position t predicts token t+1; response tokens are [prompt_len, n)
        for t in range(e["prompt_len"] - 1, n - 1):
            tgt[i, t] = ids[t + 1]
    return x.to(device), tgt.to(device)


def iterate_batches(data, batch_size, shuffle=True, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = [idx[i] for i in torch.randperm(len(data), generator=g).tolist()]
    # sort within shard by length to reduce padding, then batch
    for i in range(0, len(idx), batch_size):
        chunk = [data[j] for j in idx[i : i + batch_size]]
        yield chunk


@torch.no_grad()
def evaluate(model, data, batch_size, device, ctx, max_batches=50):
    model.eval()
    total, count = 0.0, 0
    for b, chunk in enumerate(iterate_batches(data, batch_size, shuffle=False)):
        if b >= max_batches:
            break
        x, tgt = make_batch(chunk, device)
        with ctx:
            logits, _ = model(x, targets=x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=IGNORE)
        total += loss.item()
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="pretrained checkpoint to fine-tune")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from contextlib import nullcontext
    ctx = torch.autocast("cuda", torch.bfloat16) if device == "cuda" else nullcontext()

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()
    print(f"loaded base model: {model.num_params():,} params")

    train = torch.load(args.data, weights_only=False)
    val = torch.load(args.val, weights_only=False)
    print(f"SFT data: {len(train):,} train / {len(val):,} val")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=(device == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "sft.jsonl", "a", encoding="utf-8")

    total_steps = args.epochs * math.ceil(len(train) / args.batch_size)
    print(f"total steps: {total_steps:,}")
    step = 0
    best = float("inf")
    for epoch in range(args.epochs):
        for chunk in iterate_batches(train, args.batch_size, shuffle=True, seed=epoch):
            lr = args.lr * min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            x, tgt = make_batch(chunk, device)
            opt.zero_grad(set_to_none=True)
            with ctx:
                logits, _ = model(x, targets=x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=IGNORE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e}")
                log_f.write(json.dumps({"step": step, "loss": round(loss.item(), 4)}) + "\n")
                log_f.flush()
            step += 1

        vloss = evaluate(model, val, args.batch_size, device, ctx)
        print(f"== epoch {epoch} done | val loss {vloss:.4f} ==")
        state = {"model": model.state_dict(), "config": ckpt["config"], "step": step}
        torch.save(state, out_dir / "latest.pt")
        if vloss < best:
            best = vloss
            torch.save(state, out_dir / "best.pt")

    log_f.close()
    print(f"done. best val loss {best:.4f}")


if __name__ == "__main__":
    main()
