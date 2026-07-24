import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.benchmark_cloze import build_items, split_last_sentence

STORIES = [
    "Tom had a red ball. He kicked it high. The ball landed in a tree.",
    "Lily found a cat. The cat was soft. She gave it some milk.",
    "A dog ran fast. It chased a stick. Then it came back happy.",
    "The sun was hot. Ben drank water. He felt much better.",
]


def test_split_last_sentence():
    ctx, end = split_last_sentence(STORIES[0])
    assert ctx == "Tom had a red ball. He kicked it high."
    assert end == "The ball landed in a tree."


def test_split_rejects_short():
    assert split_last_sentence("One sentence only.") is None
    assert split_last_sentence("Two sentences. Here.") is None  # <3 sentences


def test_build_items_shape_and_distinct_distractor():
    items = build_items(STORIES, n=3, seed=0)
    assert len(items) == 3
    for ctx, true_end, distractor in items:
        assert isinstance(ctx, str) and isinstance(true_end, str)
        assert true_end != distractor  # distractor comes from a different story
