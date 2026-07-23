"""Phase 5: build an instruction-tuning set from TinyStories-Instruct.

Each example becomes a (prompt, response) pair:
    prompt   = "Features: ...\nWords: ...\nSummary: ...\nStory:"
    response = " <the story><eot>"
We tokenize with the *already-trained* base tokenizer and store token ids plus
the prompt length, so SFT can mask the loss over the prompt.

Usage:
    python scripts/prepare_sft.py --max-examples 20000 --out data/sft/train.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.bpe import BPETokenizer  # noqa: E402

FIELDS = ("Features", "Words", "Summary")


def parse_instruct(text: str) -> tuple[str, str] | None:
    """TinyStories-Instruct rows interleave instruction fields and the story."""
    if "Story:" not in text:
        return None
    head, story = text.split("Story:", 1)
    story = story.strip()
    if not story:
        return None
    parts = []
    for f in FIELDS:
        m = re.search(rf"{f}:\s*(.*)", head)
        if m:
            val = m.group(1).strip()
            if val:
                parts.append(f"{f}: {val}")
    if not parts:
        return None
    prompt = "\n".join(parts) + "\nStory:"
    return prompt, " " + story


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tinystories/tokenizer.json")
    ap.add_argument("--out", default="data/sft/train.pt")
    ap.add_argument("--val-out", default="data/sft/val.pt")
    ap.add_argument("--max-examples", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    from datasets import load_dataset

    tok = BPETokenizer.load(args.tokenizer)
    print("loading TinyStories-Instruct...")
    ds = load_dataset("roneneldan/TinyStories-Instruct")

    def build(split, limit):
        examples = []
        for text in ds[split]["text"]:
            parsed = parse_instruct(text)
            if not parsed:
                continue
            prompt, response = parsed
            p_ids = tok.encode(prompt)
            r_ids = tok.encode(response) + [tok.eot_id]
            ids = p_ids + r_ids
            if len(ids) > args.max_len:
                continue
            examples.append({"ids": ids, "prompt_len": len(p_ids)})
            if len(examples) >= limit:
                break
        return examples

    train = build("train", args.max_examples)
    val = build("validation", max(200, args.max_examples // 50))
    print(f"built {len(train):,} train / {len(val):,} val examples")

    for path, data in [(args.out, train), (args.val_out, val)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
