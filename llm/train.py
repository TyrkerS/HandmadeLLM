"""Training loop, written by hand (no HF Trainer).

Features: AdamW with decay/no-decay param groups, bf16 autocast, gradient
accumulation, gradient clipping, cosine LR schedule with linear warmup,
optional gradient checkpointing and torch.compile, checkpoint/resume,
JSONL metrics + optional Weights & Biases logging.

Usage:
    python -m llm.train --config configs/tinystories_30m.yaml
    python -m llm.train --config configs/tinystories_30m.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .config import Config
from .data import get_batch, load_bin
from .model import Transformer

PTDTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def cosine_lr(step: int, *, lr: float, min_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused)


@torch.no_grad()
def estimate_loss(model, data, cfg, device, ctx) -> float:
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for i in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.batch_size, model_seq_len(model), device)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def model_seq_len(model) -> int:
    m = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    return m.cfg.max_seq_len


def raw_model(model):
    return getattr(model, "_orig_mod", model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    tcfg, mcfg = cfg.train, cfg.model

    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = PTDTYPE[tcfg.dtype]
    ctx = (
        torch.autocast(device_type="cuda", dtype=ptdtype)
        if device == "cuda" and ptdtype != torch.float32
        else nullcontext()
    )

    train_data = load_bin(tcfg.train_bin)
    val_data = load_bin(tcfg.val_bin)
    print(f"train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    model = Transformer(mcfg).to(device)
    model.grad_checkpointing = tcfg.grad_checkpointing
    print(f"model params: {model.num_params():,} ({model.num_params(non_embedding=True):,} non-embedding)")

    optimizer = build_optimizer(model, tcfg)

    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_val = float("inf")

    ckpt_path = out_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"resumed from step {start_step}")

    if tcfg.compile:
        try:
            model = torch.compile(model)
            # trigger compilation now so failures surface before the loop
            _probe = torch.zeros((1, mcfg.max_seq_len), dtype=torch.long, device=device)
            with ctx:
                model(_probe, _probe)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile unavailable ({type(e).__name__}); continuing eager. "
                  "Install triton (triton-windows on Windows) to enable it.")
            model = raw_model(model)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project=tcfg.wandb_project, name=tcfg.run_name, config=cfg.to_dict())

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = open(metrics_path, "a", encoding="utf-8")

    tokens_per_step = tcfg.batch_size * tcfg.grad_accum_steps * mcfg.max_seq_len
    print(f"effective batch: {tokens_per_step:,} tokens/step")

    model.train()
    t0 = time.time()
    for step in range(start_step, tcfg.max_steps):
        lr = cosine_lr(step, lr=tcfg.lr, min_lr=tcfg.min_lr, warmup=tcfg.warmup_steps, max_steps=tcfg.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(tcfg.grad_accum_steps):
            x, y = get_batch(train_data, tcfg.batch_size, mcfg.max_seq_len, device)
            with ctx:
                _, loss = model(x, y)
            loss_accum += loss.item()
            (loss / tcfg.grad_accum_steps).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        if step % tcfg.log_interval == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            steps_done = tcfg.log_interval if step > start_step else 1
            tok_s = tokens_per_step * steps_done / dt
            vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            loss_avg = loss_accum / tcfg.grad_accum_steps
            rec = {
                "step": step,
                "loss": round(loss_avg, 4),
                "lr": lr,
                "grad_norm": round(float(grad_norm), 3),
                "tok_per_s": round(tok_s),
                "vram_gb": round(vram, 2),
            }
            print(f"step {step:6d} | loss {loss_avg:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s | {vram:.2f} GB")
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            if run:
                run.log(rec, step=step)

        if step > 0 and step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
            val_loss = estimate_loss(model, val_data, tcfg, device, ctx)
            print(f"step {step:6d} | val loss {val_loss:.4f}")
            if run:
                run.log({"val_loss": val_loss}, step=step)
            state = {
                "model": raw_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "best_val": min(best_val, val_loss),
                "config": cfg.to_dict(),
            }
            torch.save(state, ckpt_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(state, out_dir / "best.pt")
            t0 = time.time()  # don't count eval time in tok/s

    metrics_f.close()
    if run:
        run.finish()
    print("done.")


if __name__ == "__main__":
    main()
