"""Download TinyStories, train the BPE tokenizer on a sample, tokenize to .bin.

Usage:
    python scripts/prepare_tinystories.py --vocab-size 8192 --out-dir data/tinystories
    # quick smoke test:
    python scripts/prepare_tinystories.py --max-stories 2000 --vocab-size 512 --out-dir data/smoke

Outputs: tokenizer.json, train.bin, val.bin (uint16 token streams, stories
separated by <|endoftext|>).
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.bpe import BPETokenizer  # noqa: E402

_tok: BPETokenizer | None = None


def _init_worker(tok_path: str) -> None:
    global _tok
    _tok = BPETokenizer.load(tok_path)


def _encode_batch(texts: list[str]) -> np.ndarray:
    ids: list[int] = []
    for t in texts:
        ids.extend(_tok.encode(t))
        ids.append(_tok.eot_id)
    return np.array(ids, dtype=np.uint16)


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def tokenize_split(texts: list[str], tok_path: str, out_path: Path, workers: int) -> None:
    t0 = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(tok_path,)) as pool:
        chunks = pool.map(_encode_batch, list(_batched(texts, 500)))
    ids = np.concatenate(chunks)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(out_path)
    print(f"  {out_path}: {len(ids):,} tokens ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/tinystories")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--tokenizer-sample-mb", type=int, default=30)
    ap.add_argument("--max-stories", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories")
    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]
    if args.max_stories:
        train_texts = train_texts[: args.max_stories]
        val_texts = val_texts[: max(100, args.max_stories // 50)]
    print(f"stories: {len(train_texts):,} train / {len(val_texts):,} val")

    tok_path = out_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"tokenizer exists, skipping training: {tok_path}")
    else:
        sample_bytes = args.tokenizer_sample_mb * 1024 * 1024
        sample, size = [], 0
        for t in train_texts:
            sample.append(t)
            size += len(t)
            if size >= sample_bytes:
                break
        print(f"training BPE (vocab {args.vocab_size}) on {size / 1e6:.1f} MB...")
        t0 = time.time()
        tok = BPETokenizer()
        tok.train("\n".join(sample), vocab_size=args.vocab_size, verbose=True)
        tok.save(tok_path)
        print(f"tokenizer trained in {time.time() - t0:.0f}s -> {tok_path}")

    print("tokenizing splits...")
    tokenize_split(val_texts, str(tok_path), out_dir / "val.bin", args.workers)
    tokenize_split(train_texts, str(tok_path), out_dir / "train.bin", args.workers)
    print("done.")


if __name__ == "__main__":
    main()
