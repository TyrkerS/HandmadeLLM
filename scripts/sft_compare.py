"""Phase 5: run fixed instruction prompts through a checkpoint (before/after SFT).

Usage:
    python scripts/sft_compare.py --ckpt checkpoints/tinystories_30m/best.pt --label before
    python scripts/sft_compare.py --ckpt checkpoints/sft_30m/best.pt --label after
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.bpe import BPETokenizer  # noqa: E402
from llm.config import ModelConfig, _build  # noqa: E402
from llm.model import Transformer  # noqa: E402

PROMPTS = [
    "Features: Dialogue\nWords: dog, jump, happy\nSummary: A boy and his dog play in the yard.\nStory:",
    "Features: MoralValue\nWords: share, cookie, friend\nSummary: Two friends learn to share.\nStory:",
    "Words: boat, river, brave\nSummary: A little girl sails a boat down the river.\nStory:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-new", type=int, default=150)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(ckpt["config"]["train"]["tokenizer_path"])

    print(f"\n########## {args.label.upper()} — {args.ckpt} ##########")
    for p in PROMPTS:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = model.generate(ids, max_new_tokens=args.max_new, temperature=0.7, top_k=50, eos_id=tok.eot_id)
        gen = tok.decode(out[0].tolist())
        print("\n--- PROMPT ---")
        print(p)
        print("--- CONTINUATION ---")
        print(gen[len(p):].strip()[:600])


if __name__ == "__main__":
    main()
