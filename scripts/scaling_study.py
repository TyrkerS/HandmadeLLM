"""Phase 3: mini scaling-law study.

Trains several model sizes under a *fixed token budget* (data-controlled) on
TinyStories, records final validation loss, and plots loss vs non-embedding
parameters. This is the pedagogical "loss goes down predictably with size"
curve — honest about being data-controlled rather than a full compute-optimal
Chinchilla sweep (noted in the writeup).

Writes versioned configs to configs/scaling/, checkpoints to
checkpoints/scaling/<name>/, a results table to samples/scaling.md, and an SVG
plot to samples/scaling.svg.

Usage:
    python scripts/scaling_study.py                 # train all sizes + plot
    python scripts/scaling_study.py --plot-only      # just regenerate table/plot
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.config import ModelConfig  # noqa: E402
from llm.model import Transformer  # noqa: E402

# (name, dim, n_layers, n_heads, n_kv_heads)
SIZES = [
    ("s1_tiny", 256, 4, 4, 2),
    ("s2_small", 384, 6, 6, 3),
    ("s3_med", 512, 8, 8, 4),
    ("s4_large", 640, 10, 10, 5),
]

# Shared fixed budget (same tokens for every size).
SHARED = dict(
    vocab_size=8192,
    max_seq_len=512,
    batch_size=24,
    grad_accum_steps=6,     # 24*6*512 = 73,728 tokens/step
    max_steps=1500,         # ~110M tokens total, identical across sizes
    lr=6.0e-4,
    min_lr=6.0e-5,
    warmup_steps=150,
    dtype="bfloat16",
    compile=True,
    eval_interval=250,
    eval_iters=50,
    log_interval=25,
)


def make_config(name, dim, n_layers, n_heads, n_kv_heads) -> Path:
    cfg = {
        "model": {
            "dim": dim, "n_layers": n_layers, "n_heads": n_heads,
            "n_kv_heads": n_kv_heads, "vocab_size": SHARED["vocab_size"],
            "max_seq_len": SHARED["max_seq_len"],
        },
        "train": {
            "train_bin": "data/tinystories/train.bin",
            "val_bin": "data/tinystories/val.bin",
            "tokenizer_path": "data/tinystories/tokenizer.json",
            "batch_size": SHARED["batch_size"], "grad_accum_steps": SHARED["grad_accum_steps"],
            "max_steps": SHARED["max_steps"], "lr": SHARED["lr"], "min_lr": SHARED["min_lr"],
            "warmup_steps": SHARED["warmup_steps"], "dtype": SHARED["dtype"],
            "compile": SHARED["compile"], "out_dir": f"checkpoints/scaling/{name}",
            "eval_interval": SHARED["eval_interval"], "eval_iters": SHARED["eval_iters"],
            "log_interval": SHARED["log_interval"], "run_name": f"scaling_{name}",
        },
    }
    path = ROOT / "configs" / "scaling" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def param_count(dim, n_layers, n_heads, n_kv_heads) -> int:
    mcfg = ModelConfig(dim=dim, n_layers=n_layers, n_heads=n_heads,
                       n_kv_heads=n_kv_heads, vocab_size=SHARED["vocab_size"],
                       max_seq_len=SHARED["max_seq_len"])
    return Transformer(mcfg).num_params(non_embedding=True)


def final_val_loss(name) -> float | None:
    # take the last checkpoint's best_val via metrics; simplest: re-eval not needed,
    # the training printed val loss — read the最 recent val from stdout log file.
    log = ROOT / "checkpoints" / "scaling" / name / "val.jsonl"
    if log.exists():
        lines = log.read_text().splitlines()
        if lines:
            return json.loads(lines[-1])["val_loss"]
    return None


def collect_and_plot():
    rows = []
    for name, dim, L, h, kv in SIZES:
        n = param_count(dim, L, h, kv)
        vl = final_val_loss(name)
        rows.append((name, n, vl))

    # table
    md = ["# Scaling study (fixed ~110M-token budget, TinyStories)\n",
          "| model | dim×layers | non-emb params | final val loss |",
          "|---|---|---|---|"]
    for (name, dim, L, h, kv), (_, n, vl) in zip(SIZES, rows):
        vls = f"{vl:.3f}" if vl is not None else "—"
        md.append(f"| {name} | {dim}×{L} | {n/1e6:.1f}M | {vls} |")
    (ROOT / "samples" / "scaling.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # SVG: log-x params vs loss
    pts = [(n, vl) for _, n, vl in rows if vl is not None]
    if len(pts) >= 2:
        W, H, pad = 640, 400, 60
        xs = [math.log10(n) for n, _ in pts]
        ys = [vl for _, vl in pts]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys) - 0.05, max(ys) + 0.05

        def sx(x): return pad + (x - x0) / (x1 - x0) * (W - 2 * pad)
        def sy(y): return H - pad - (y - y0) / (y1 - y0) * (H - 2 * pad)

        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="#2563eb"/>'
                       f'<text x="{sx(x):.1f}" y="{sy(y)-12:.1f}" font-size="11" fill="#374151" text-anchor="middle">{y:.2f}</text>'
                       for x, y in zip(xs, ys))
        labels = "".join(f'<text x="{sx(math.log10(n)):.1f}" y="{H-pad+20}" font-size="11" fill="#6b7280" text-anchor="middle">{n/1e6:.0f}M</text>'
                         for n, _ in pts)
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{W/2}" y="26" font-size="15" font-weight="600" fill="#111827" text-anchor="middle">Mini scaling law — TinyStories, fixed token budget</text>
<polyline points="{poly}" fill="none" stroke="#93c5fd" stroke-width="2"/>
{dots}{labels}
<text x="{W/2}" y="{H-16}" font-size="12" fill="#6b7280" text-anchor="middle">non-embedding parameters (log scale)</text>
<text x="18" y="{H/2}" font-size="12" fill="#6b7280" text-anchor="middle" transform="rotate(-90 18 {H/2})">final val loss</text>
</svg>'''
        (ROOT / "samples" / "scaling.svg").write_text(svg, encoding="utf-8")
    print("wrote samples/scaling.md and samples/scaling.svg")
    for name, n, vl in rows:
        print(f"  {name}: {n/1e6:.1f}M params, val loss {vl if vl else 'pending'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        for name, dim, L, h, kv in SIZES:
            cfg_path = make_config(name, dim, L, h, kv)
            print(f"\n=== training {name} ({param_count(dim,L,h,kv)/1e6:.1f}M non-emb) ===")
            subprocess.run([sys.executable, "-m", "llm.train", "--config", str(cfg_path)], check=True, cwd=ROOT)

    collect_and_plot()


if __name__ == "__main__":
    main()
