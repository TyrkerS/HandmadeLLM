"""Phase 0 definition-of-done: a ~10M byte-level model overfits one tiny batch.

If loss doesn't collapse toward 0, something in the stack (model, loss,
optimizer, device) is broken — better to find out before scaling up.

Usage:
    python scripts/overfit_sanity.py            # 500 steps, asserts loss < 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.config import ModelConfig  # noqa: E402
from llm.model import Transformer  # noqa: E402

TEXT = (
    "The little robot walked through the quiet valley, counting stones as it went. "
    "One stone, two stones, three stones - each one a small memory of the mountain "
    "that once stood here. When the rain came, the robot hid under an old bridge "
    "and listened to the water telling stories about the sea it had never seen. "
) * 64


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=0.1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    cfg = ModelConfig(
        dim=384, n_layers=6, n_heads=6, n_kv_heads=3,
        vocab_size=256, max_seq_len=256,
    )
    model = Transformer(cfg).to(device)
    print(f"device: {device} | params: {model.num_params():,}")

    data = torch.tensor(list(TEXT.encode("utf-8")), dtype=torch.long)
    ix = torch.randint(len(data) - 257, (32,))
    x = torch.stack([data[i : i + 256] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 257] for i in ix]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    ctx = torch.autocast("cuda", torch.bfloat16) if device == "cuda" else nullcontext()

    t0 = time.time()
    loss = None
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / 20)  # short warmup
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with ctx:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = 32 * 256 * args.steps / dt
    final = loss.item()
    print(f"\nfinal loss: {final:.4f} | {tok_s:,.0f} tok/s | {dt:.1f}s")
    if device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    assert final < args.threshold, f"FAIL: loss {final:.4f} >= {args.threshold}"
    print(f"PASS: loss {final:.4f} < {args.threshold} - the training stack works.")


if __name__ == "__main__":
    main()
