"""Phase 6: a bespoke downstream benchmark — Story Cloze.

HellaSwag-style multiple choice is too hard for a TinyStories-scale model, so I
built a benchmark matched to the domain and defensible on its own terms: given
a story with its last sentence removed, does the model assign higher average
log-probability to the TRUE ending than to a distractor ending sampled from a
different story? Accuracy = fraction of items where the true ending wins.

This tests narrative coherence (not just fluency): a good distractor is a real,
fluent TinyStories sentence — it's only wrong *in context*. Chance = 50%.

The benchmark is built deterministically from the held-out split, so it's
reproducible and never seen in training.

Usage:
    python -m llm.benchmark_cloze --ckpt checkpoints/tinystories_30m/best.pt \
        --tokenizer data/tinystories/tokenizer.json --n 500
"""

from __future__ import annotations

import argparse
import random
import re

import torch
import torch.nn.functional as F

from .bpe import BPETokenizer
from .config import ModelConfig, _build
from .model import Transformer

_SENT = re.compile(r"(?<=[.!?])\s+")


def split_last_sentence(story: str) -> tuple[str, str] | None:
    sents = [s.strip() for s in _SENT.split(story.strip()) if s.strip()]
    if len(sents) < 3:
        return None
    context = " ".join(sents[:-1])
    ending = sents[-1]
    if len(ending) < 8:
        return None
    return context, ending


def build_items(stories, n, seed=0):
    rng = random.Random(seed)
    parsed = []
    for s in stories:
        sp = split_last_sentence(s)
        if sp:
            parsed.append(sp)
        if len(parsed) >= n * 2:
            break
    items = []
    for i, (ctx, true_end) in enumerate(parsed):
        if len(items) >= n:
            break
        # distractor: an ending from a different story
        j = rng.randrange(len(parsed))
        while j == i:
            j = rng.randrange(len(parsed))
        distractor = parsed[j][1]
        items.append((ctx, true_end, distractor))
    return items


@torch.no_grad()
def avg_logprob(model, tok, context: str, ending: str, device) -> float:
    ctx_ids = tok.encode(context + " ")
    end_ids = tok.encode(ending)
    if not end_ids:
        return -1e9
    ids = ctx_ids + end_ids
    ids = ids[-model.cfg.max_seq_len:]
    x = torch.tensor([ids], device=device)
    logits, _ = model(x, targets=x)
    logp = F.log_softmax(logits.float(), dim=-1)[0]
    # score only the ending tokens
    n_end = len(end_ids)
    total = 0.0
    for pos in range(len(ids) - n_end, len(ids)):
        if pos == 0:
            continue
        total += logp[pos - 1, ids[pos]].item()
    return total / n_end


def evaluate(model, tok, items, device) -> float:
    correct = 0
    for ctx, true_end, distractor in items:
        lp_true = avg_logprob(model, tok, ctx, true_end, device)
        lp_dist = avg_logprob(model, tok, ctx, distractor, device)
        correct += lp_true > lp_dist
    return correct / len(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(args.tokenizer or ckpt["config"]["train"]["tokenizer_path"])

    val = load_dataset("roneneldan/TinyStories")["validation"]["text"]
    items = build_items(val, args.n)
    acc = evaluate(model, tok, items, device)
    print(f"Story-Cloze accuracy ({len(items)} items, chance=0.50): {acc:.3f}")


if __name__ == "__main__":
    main()
