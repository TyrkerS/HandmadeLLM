import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm.sft import IGNORE, make_batch


def test_prompt_tokens_are_masked():
    # ids = [p0, p1, p2, r0, r1]  prompt_len=3  -> only response predictions supervised
    ex = [{"ids": [10, 11, 12, 20, 21], "prompt_len": 3}]
    x, tgt = make_batch(ex, "cpu")
    # positions predicting a prompt token (t < prompt_len-1) are ignored
    assert tgt[0, 0].item() == IGNORE  # predicts p1 (prompt) -> masked
    assert tgt[0, 1].item() == IGNORE  # predicts p2 (prompt) -> masked
    # position prompt_len-1=2 predicts r0 (first response token) -> supervised
    assert tgt[0, 2].item() == 20
    assert tgt[0, 3].item() == 21
    # last position has no next token
    assert tgt[0, 4].item() == IGNORE


def test_right_padding_masked():
    ex = [
        {"ids": [1, 2, 3, 4], "prompt_len": 2},
        {"ids": [5, 6, 7], "prompt_len": 1},
    ]
    x, tgt = make_batch(ex, "cpu")
    assert x.shape == (2, 4)
    # shorter example padded with zeros in x
    assert x[1, 3].item() == 0
    # pad region contributes no supervised targets
    assert tgt[1, 3].item() == IGNORE
    # supervised counts = response_len per example
    supervised = (tgt != IGNORE).sum(dim=1)
    assert supervised[0].item() == 2  # r=[3,4] -> 2 targets
    assert supervised[1].item() == 2  # r=[6,7] -> 2 targets
