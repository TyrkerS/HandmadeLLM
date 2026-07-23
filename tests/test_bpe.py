import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from llm.bpe import BPETokenizer

CORPUS = (
    "Once upon a time there was a little girl named Lily. "
    "Lily loved to play in the garden with her dog. "
    "One day, the dog found a big red ball under the tree. "
) * 50


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    return t


def test_vocab_size(tok):
    # vocab_size is an upper bound: training stops early once no pair has
    # frequency >= 2 (this tiny corpus is highly repetitive, so it does).
    assert 256 + 1 < tok.vocab_size <= 400
    assert len(tok.merges) == tok.vocab_size - 256 - 1  # 1 special


@pytest.mark.parametrize(
    "text",
    [
        "Once upon a time there was a little girl.",
        "hello world",
        "  multiple   spaces\nand\nnewlines\t tabs",
        "números y acentos: canción, ñoño, 1234!",
        "emoji test 🚀🔥 done",
        "",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_compression(tok):
    ids = tok.encode(CORPUS)
    assert len(ids) < len(CORPUS.encode("utf-8"))  # merges actually compress


def test_merges_used(tok):
    ids = tok.encode("Once upon a time")
    assert any(i >= 256 for i in ids)


def test_eot(tok):
    assert tok.eot_id == tok.vocab_size - 1
    assert tok.decode([tok.eot_id]) == "<|endoftext|>"


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    text = "Lily loved to play in the garden."
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.vocab_size == tok.vocab_size
