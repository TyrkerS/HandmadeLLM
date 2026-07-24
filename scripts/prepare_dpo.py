"""Phase 5 (stretch): build a preference dataset for DPO.

Each preference pair is (prompt, chosen, rejected):
    chosen   = the real TinyStories-Instruct story (follows the instruction)
    rejected = a generation from the *base* (pre-SFT) model for the same prompt
               (it rambles / ignores the required words)

DPO then teaches the policy to prefer chosen-like over rejected-like responses.
Reusing the SFT dataset gives us (prompt, chosen) for free; we only need to
generate the rejected completions.

Usage:
    python scripts/prepare_dpo.py --base checkpoints/tinystories_30m/best.pt \
        --sft-data data/sft/train.pt --out data/dpo/train.pt --n 2500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.config import ModelConfig, _build  # noqa: E402
from llm.model import Transformer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pre-SFT base checkpoint (source of rejected)")
    ap.add_argument("--sft-data", default="data/sft/train.pt")
    ap.add_argument("--out", default="data/dpo/train.pt")
    ap.add_argument("--val-out", default="data/dpo/val.pt")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--max-new", type=int, default=180)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    ckpt = torch.load(args.base, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    eot = None
    # the eot id is the last special token; read it from the tokenizer
    from llm.bpe import BPETokenizer
    tok = BPETokenizer.load(ckpt["config"]["train"]["tokenizer_path"])
    eot = tok.eot_id

    sft = torch.load(args.sft_data, weights_only=False)[: args.n]
    pairs = []
    for i, ex in enumerate(sft):
        ids = ex["ids"]
        plen = ex["prompt_len"]
        prompt_ids = ids[:plen]
        chosen = ids[plen:]  # real response (incl. trailing eot)

        # rejected: base model's own continuation of the same prompt
        pt = torch.tensor([prompt_ids], device=device)
        with torch.inference_mode():
            out = model.generate(pt, max_new_tokens=args.max_new, temperature=0.9,
                                 top_k=50, eos_id=eot)
        rejected = out[0].tolist()[plen:]
        if not rejected:
            continue
        pairs.append({"prompt_ids": prompt_ids, "chosen": chosen, "rejected": rejected})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(sft)} pairs")

    n_val = max(100, len(pairs) // 25)
    train, val = pairs[:-n_val], pairs[-n_val:]
    for path, data in [(args.out, train), (args.val_out, val)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        print(f"wrote {path}: {len(data)} pairs")


if __name__ == "__main__":
    main()
