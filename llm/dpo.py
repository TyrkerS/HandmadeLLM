"""Phase 5 (stretch): Direct Preference Optimization from scratch.

DPO aligns a policy to preference pairs *without* a separate reward model or RL.
Given (prompt, chosen, rejected), with a frozen reference model (the SFT model),
the loss is:

    L = -log sigmoid( beta * [ (logp_pol(chosen) - logp_ref(chosen))
                             - (logp_pol(rejected) - logp_ref(rejected)) ] )

where logp(·) is the sum of token log-probs over the *response* tokens only.
Minimizing L raises the policy's relative preference for chosen over rejected,
while the reference term (via beta) keeps it from drifting too far.

Usage:
    python -m llm.dpo --init checkpoints/sft_30m/best.pt \
        --data data/dpo/train.pt --val data/dpo/val.pt --out checkpoints/dpo_30m
"""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig, _build
from .model import Transformer

IGNORE = -100


def _pad(seqs, pad_val, device):
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_val, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out.to(device)


def build_batch(pairs, device):
    """Concatenate prompt+response for chosen and rejected; mask everything but
    the response tokens. Returns dicts for chosen and rejected."""
    def side(key):
        ids, tgt = [], []
        for p in pairs:
            full = p["prompt_ids"] + p[key]
            plen = len(p["prompt_ids"])
            t = [IGNORE] * len(full)
            for i in range(plen - 1, len(full) - 1):
                t[i] = full[i + 1]
            ids.append(full)
            tgt.append(t)
        return _pad(ids, 0, device), _pad(tgt, IGNORE, device)

    x_c, t_c = side("chosen")
    x_r, t_r = side("rejected")
    return (x_c, t_c), (x_r, t_r)


def seq_logprob(model, x, tgt):
    """Sum of log-probs of the target (response) tokens for each sequence."""
    logits, _ = model(x, targets=x)  # full logits (B,T,V)
    logp = F.log_softmax(logits.float(), dim=-1)
    mask = tgt != IGNORE
    safe = tgt.clamp_min(0)
    tok_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(dim=1)  # (B,)


def dpo_loss(policy, ref, chosen, rejected, beta):
    (x_c, t_c), (x_r, t_r) = chosen, rejected
    pol_c = seq_logprob(policy, x_c, t_c)
    pol_r = seq_logprob(policy, x_r, t_r)
    with torch.no_grad():
        ref_c = seq_logprob(ref, x_c, t_c)
        ref_r = seq_logprob(ref, x_r, t_r)
    logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean()  # fraction where chosen is preferred
    margin = (pol_c - pol_r).mean()
    return loss, acc, margin


def iterate(data, bs, shuffle=True, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = [idx[i] for i in torch.randperm(len(data), generator=g).tolist()]
    for i in range(0, len(idx), bs):
        yield [data[j] for j in idx[i : i + bs]]


@torch.no_grad()
def evaluate(policy, ref, data, bs, device, beta, max_batches=30):
    policy.eval()
    accs = []
    for b, chunk in enumerate(iterate(data, bs, shuffle=False)):
        if b >= max_batches:
            break
        c, r = build_batch(chunk, device)
        _, acc, _ = dpo_loss(policy, ref, c, r, beta)
        accs.append(acc.item())
    policy.train()
    return sum(accs) / max(1, len(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="SFT checkpoint = policy init + frozen reference")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = torch.autocast("cuda", torch.bfloat16) if device == "cuda" else nullcontext()

    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])

    policy = Transformer(mcfg).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.train()

    ref = Transformer(mcfg).to(device)
    ref.load_state_dict(ckpt["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    train = torch.load(args.data, weights_only=False)
    val = torch.load(args.val, weights_only=False)
    print(f"DPO pairs: {len(train):,} train / {len(val):,} val | beta={args.beta}")

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=(device == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "dpo.jsonl", "a", encoding="utf-8")

    print(f"init preference accuracy (val): {evaluate(policy, ref, val, args.batch_size, device, args.beta):.3f}")
    step = 0
    for epoch in range(args.epochs):
        for chunk in iterate(train, args.batch_size, shuffle=True, seed=epoch):
            lr = args.lr * min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            c, r = build_batch(chunk, device)
            opt.zero_grad(set_to_none=True)
            with ctx:
                loss, acc, margin = dpo_loss(policy, ref, c, r, args.beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if step % 25 == 0:
                print(f"epoch {epoch} step {step:4d} | loss {loss.item():.4f} | "
                      f"pref_acc {acc.item():.3f} | margin {margin.item():+.3f}")
                log_f.write(json.dumps({"step": step, "loss": round(loss.item(), 4),
                                        "pref_acc": round(acc.item(), 3)}) + "\n")
                log_f.flush()
            step += 1

    val_acc = evaluate(policy, ref, val, args.batch_size, device, args.beta)
    print(f"final preference accuracy (val): {val_acc:.3f}")
    torch.save({"model": policy.state_dict(), "config": ckpt["config"], "step": step},
               out_dir / "best.pt")
    log_f.close()
    print(f"done -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
