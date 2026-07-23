"""Phase 6: lightweight generation-quality rubric (reference-free heuristics).

Not a substitute for human/LLM judging — a cheap, reproducible signal to track
regressions across checkpoints. Scores each generation 0–1 on:

  fluency        : fraction of alphabetic tokens that are dictionary-plausible
                   (here: in the training-corpus vocabulary the model saw)
  non_repetition : 1 - fraction of repeated 4-grams (penalizes loops)
  completion     : ends on sentence punctuation (. ! ?)
  diversity      : distinct-2 (unique bigrams / total bigrams)

Usage:
    python -m llm.rubric --ckpt checkpoints/tinystories_30m/best.pt
"""

from __future__ import annotations

import argparse
import re

import torch

from .bpe import BPETokenizer
from .config import ModelConfig, _build
from .model import Transformer

PROMPTS = [
    "Once upon a time",
    "The little cat wanted to",
    "One day, Tom and Lily went to the park and",
    "The wizard opened the old book and",
]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def score_text(text: str, known_words: set[str]) -> dict:
    words = _words(text)
    if not words:
        return dict(fluency=0.0, non_repetition=0.0, completion=0.0, diversity=0.0)

    fluency = sum(w in known_words for w in words) / len(words)

    grams4 = list(zip(words, words[1:], words[2:], words[3:]))
    non_rep = 1.0 if not grams4 else len(set(grams4)) / len(grams4)

    completion = 1.0 if text.strip().endswith((".", "!", "?")) else 0.0

    bigrams = list(zip(words, words[1:]))
    diversity = 0.0 if not bigrams else len(set(bigrams)) / len(bigrams)

    return dict(fluency=round(fluency, 3), non_repetition=round(non_rep, 3),
                completion=completion, diversity=round(diversity, 3))


def build_vocab(tok: BPETokenizer) -> set[str]:
    """Approximate 'known words' from the tokenizer's merged pieces."""
    words = set()
    for tid in range(tok.vocab_size):
        try:
            piece = tok.vocab[tid].decode("utf-8", errors="ignore")
        except Exception:
            continue
        for w in _words(piece):
            words.add(w)
    return words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(ckpt["config"]["train"]["tokenizer_path"])
    known = build_vocab(tok)

    print("| prompt | fluency | non-rep | complete | distinct-2 |")
    print("|---|---|---|---|---|")
    agg = dict(fluency=0.0, non_repetition=0.0, completion=0.0, diversity=0.0)
    for p in PROMPTS:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = model.generate(ids, max_new_tokens=args.max_new, temperature=args.temperature,
                             top_k=50, eos_id=tok.eot_id)
        text = tok.decode(out[0].tolist())
        s = score_text(text, known)
        for k in agg:
            agg[k] += s[k] / len(PROMPTS)
        print(f"| {p[:28]:<28} | {s['fluency']} | {s['non_repetition']} | {s['completion']:.0f} | {s['diversity']} |")
    print(f"| **mean** | {agg['fluency']:.3f} | {agg['non_repetition']:.3f} | "
          f"{agg['completion']:.2f} | {agg['diversity']:.3f} |")


if __name__ == "__main__":
    main()
