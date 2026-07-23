"""Byte-level BPE tokenizer, implemented from scratch.

Training uses the classic word-frequency formulation with incremental pair
counts: the corpus is pre-tokenized into "words" with a regex, each unique
word is a sequence of byte tokens, and on every merge we only touch the words
that actually contain the merged pair. This keeps training on tens of MB of
text in the seconds-to-minutes range in pure Python.

Token id layout:
    [0, 255]                      raw bytes
    [256, 256 + n_merges)         merge tokens, in merge order
    [256 + n_merges, vocab_size)  special tokens
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

# Simplified GPT-2-style pre-tokenization pattern (stdlib `re`, ASCII classes).
# Good enough for English corpora like TinyStories; documented limitation.
_PAT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

_DEFAULT_SPECIALS = ("<|endoftext|>",)


def _merge_word(word: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    n = len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        specials: tuple[str, ...] = _DEFAULT_SPECIALS,
    ):
        self.merges: list[tuple[int, int]] = merges or []
        self.specials: tuple[str, ...] = tuple(specials)
        self._rebuild()

    # ------------------------------------------------------------- vocab

    def _rebuild(self) -> None:
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special_to_id: dict[str, int] = {}
        base = 256 + len(self.merges)
        for j, s in enumerate(self.specials):
            self.special_to_id[s] = base + j
            self.vocab[base + j] = s.encode("utf-8")
        self._word_cache: dict[bytes, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.specials)

    @property
    def eot_id(self) -> int:
        return self.special_to_id["<|endoftext|>"]

    # ------------------------------------------------------------- train

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        n_merges = vocab_size - 256 - len(self.specials)
        if n_merges <= 0:
            raise ValueError(f"vocab_size={vocab_size} leaves no room for merges")

        # word frequencies over the pre-tokenized corpus
        freqs: dict[str, int] = defaultdict(int)
        for w in _PAT.findall(text):
            freqs[w] += 1
        words: list[list[int]] = [list(w.encode("utf-8")) for w in freqs]
        wfreq: list[int] = list(freqs.values())

        # initial pair counts + reverse index pair -> word indices
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, word in enumerate(words):
            f = wfreq[wi]
            for p in zip(word, word[1:]):
                pair_counts[p] += f
                pair_where[p].add(wi)

        self.merges = []
        for step in range(n_merges):
            if not pair_counts:
                break
            # deterministic tie-break: highest count, then lowest pair ids
            best = max(pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))
            pair, count = best
            if count < 2:
                break
            new_id = 256 + step
            self.merges.append(pair)

            for wi in list(pair_where.get(pair, ())):
                old = words[wi]
                f = wfreq[wi]
                new = _merge_word(old, pair, new_id)
                if new == old:  # stale index (lazy cleanup), nothing to do
                    continue
                for p in zip(old, old[1:]):
                    c = pair_counts.get(p)
                    if c is not None:
                        c -= f
                        if c <= 0:
                            del pair_counts[p]
                        else:
                            pair_counts[p] = c
                words[wi] = new
                for p in zip(new, new[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    pair_where[p].add(wi)

            pair_counts.pop(pair, None)
            pair_where.pop(pair, None)

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{n_merges}  {pair} (count {count})")

        self._rebuild()

    # ---------------------------------------------------- encode / decode

    def _encode_word(self, wb: bytes) -> list[int]:
        cached = self._word_cache.get(wb)
        if cached is not None:
            return cached
        parts = list(wb)
        while len(parts) >= 2:
            pairs = set(zip(parts, parts[1:]))
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            rank = self.ranks.get(best)
            if rank is None:
                break
            parts = _merge_word(parts, best, 256 + rank)
        self._word_cache[wb] = parts
        return parts

    def encode(self, text: str) -> list[int]:
        """Encode plain text (special tokens in text are NOT parsed as specials)."""
        ids: list[int] = []
        for w in _PAT.findall(text):
            ids.extend(self._encode_word(w.encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persist

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "specials": list(self.specials),
            "merges": [[a, b] for a, b in self.merges],
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            merges=[tuple(p) for p in payload["merges"]],
            specials=tuple(payload["specials"]),
        )
