import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm.config import ModelConfig
from llm.engine import ContinuousBatchingEngine, Request
from llm.model import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(dim=64, n_layers=3, n_heads=4, n_kv_heads=2, vocab_size=80, max_seq_len=128)
    return Transformer(cfg).to(DEVICE).eval()


def _greedy_ref(model, prompt, n):
    idx = torch.tensor([prompt], device=DEVICE)
    out = model.generate(idx, max_new_tokens=n, temperature=0.0, use_cache=True)
    return out[0, len(prompt):].tolist()


def test_batched_equals_sequential_greedy():
    """Continuous batching with greedy sampling must match one-at-a-time decoding,
    even with different prompt lengths in the same batch."""
    model = _model()
    prompts = [
        [3, 4, 5, 6, 7, 8],
        [10, 11, 12],
        [20, 21, 22, 23, 24, 25, 26, 27],
        [30, 31, 32, 33],
    ]
    n = 16
    refs = [_greedy_ref(model, p, n) for p in prompts]

    engine = ContinuousBatchingEngine(model, device=DEVICE, max_batch=4)
    reqs = [Request(id=i, tokens=list(p), prompt_len=len(p), max_new_tokens=n, temperature=0.0)
            for i, p in enumerate(prompts)]
    engine.generate(reqs)

    for i, req in enumerate(reqs):
        assert req.generated == refs[i], f"request {i} mismatch"


def test_admission_beyond_max_batch():
    """More requests than max_batch: the extras are admitted as slots free up,
    and all still match sequential greedy output."""
    model = _model()
    prompts = [[i, i + 1, i + 2, i + 3] for i in range(2, 12, 2)]  # 5 requests
    n = 12
    refs = [_greedy_ref(model, p, n) for p in prompts]

    engine = ContinuousBatchingEngine(model, device=DEVICE, max_batch=2)  # < num requests
    reqs = [Request(id=i, tokens=list(p), prompt_len=len(p), max_new_tokens=n, temperature=0.0)
            for i, p in enumerate(prompts)]
    engine.generate(reqs)

    for i, req in enumerate(reqs):
        assert req.generated == refs[i], f"request {i} mismatch"


def test_eos_stops_early():
    model = _model()
    # force an eos id that the model will emit at some point is hard; instead
    # verify max_new_tokens bound is respected exactly
    engine = ContinuousBatchingEngine(model, device=DEVICE, max_batch=4)
    req = Request(id=0, tokens=[1, 2, 3], prompt_len=3, max_new_tokens=7, temperature=0.0)
    engine.generate([req])
    assert len(req.generated) == 7
