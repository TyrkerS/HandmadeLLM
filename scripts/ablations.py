"""Phase 6: architecture ablations — numbers, not opinions.

Trains a fixed small model under one budget, flipping a single architectural
choice at a time, and tabulates final validation loss:

    baseline : RoPE + SwiGLU + RMSNorm + GQA (the modern stack)
    variants : learned pos-emb | GELU MLP | LayerNorm | MHA (no GQA)

Writes versioned configs to configs/ablations/, checkpoints under
checkpoints/ablations/<name>/, and a results table to samples/ablations.md.

Usage:
    python scripts/ablations.py               # train all, then tabulate
    python scripts/ablations.py --table-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# base architecture (small, trains fast); each variant overrides model fields
BASE_MODEL = dict(dim=384, n_layers=6, n_heads=6, n_kv_heads=3,
                  vocab_size=8192, max_seq_len=512)

VARIANTS = {
    "baseline_rope_swiglu_rmsnorm_gqa": {},
    "learned_pos": {"pos_emb": "learned"},
    "gelu_mlp": {"mlp": "gelu"},
    "layernorm": {"norm": "layernorm"},
    "mha_no_gqa": {"n_kv_heads": 6},
}

TRAIN = dict(
    train_bin="data/tinystories/train.bin", val_bin="data/tinystories/val.bin",
    tokenizer_path="data/tinystories/tokenizer.json",
    batch_size=24, grad_accum_steps=6, max_steps=1200,
    lr=6.0e-4, min_lr=6.0e-5, warmup_steps=150, dtype="bfloat16", compile=True,
    eval_interval=200, eval_iters=50, log_interval=25,
)


def make_config(name, overrides) -> Path:
    model = {**BASE_MODEL, **overrides}
    train = {**TRAIN, "out_dir": f"checkpoints/ablations/{name}", "run_name": f"abl_{name}"}
    path = ROOT / "configs" / "ablations" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"model": model, "train": train}, sort_keys=False), encoding="utf-8")
    return path


def final_val(name) -> float | None:
    f = ROOT / "checkpoints" / "ablations" / name / "val.jsonl"
    if f.exists() and f.read_text().strip():
        return json.loads(f.read_text().splitlines()[-1])["val_loss"]
    return None


def tabulate():
    base = final_val("baseline_rope_swiglu_rmsnorm_gqa")
    md = ["# Architecture ablations (TinyStories, fixed budget)\n",
          "Single-change-at-a-time vs the modern baseline (RoPE + SwiGLU + RMSNorm + GQA).\n",
          "| variant | change | final val loss | Δ vs baseline |",
          "|---|---|---|---|"]
    change = {
        "baseline_rope_swiglu_rmsnorm_gqa": "— (baseline)",
        "learned_pos": "RoPE → learned pos-emb",
        "gelu_mlp": "SwiGLU → GELU MLP",
        "layernorm": "RMSNorm → LayerNorm",
        "mha_no_gqa": "GQA → full MHA",
    }
    for name in VARIANTS:
        vl = final_val(name)
        vls = f"{vl:.3f}" if vl is not None else "—"
        if vl is not None and base is not None and name != "baseline_rope_swiglu_rmsnorm_gqa":
            delta = f"{vl - base:+.3f}"
        else:
            delta = "—"
        md.append(f"| {name} | {change[name]} | {vls} | {delta} |")
    (ROOT / "samples").mkdir(exist_ok=True)
    (ROOT / "samples" / "ablations.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if not args.table_only:
        for name, ov in VARIANTS.items():
            cfg = make_config(name, ov)
            print(f"\n=== training {name} ===")
            subprocess.run([sys.executable, "-m", "llm.train", "--config", str(cfg)], check=True, cwd=ROOT)

    tabulate()


if __name__ == "__main__":
    main()
