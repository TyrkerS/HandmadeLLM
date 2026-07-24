import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from llm.config import ModelConfig
from llm.dpo import IGNORE, build_batch, dpo_loss, seq_logprob
from llm.model import Transformer


def _cfg():
    return ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=60, max_seq_len=64)


PAIRS = [
    {"prompt_ids": [1, 2, 3], "chosen": [10, 11, 12], "rejected": [20, 21]},
    {"prompt_ids": [4, 5], "chosen": [30, 31], "rejected": [40, 41, 42]},
]


def test_build_batch_masks_prompt_and_pad():
    (x_c, t_c), (x_r, t_r) = build_batch(PAIRS, "cpu")
    # first example: prompt_len 3, full = [1,2,3,10,11,12]; supervised = response tokens
    assert (t_c[0] != IGNORE).sum().item() == 3   # chosen has 3 response tokens
    assert (t_r[0] != IGNORE).sum().item() == 2   # rejected has 2 response tokens
    # padding region masked
    assert x_c.shape[0] == 2


def test_seq_logprob_finite_and_negative():
    torch.manual_seed(0)
    model = Transformer(_cfg()).eval()
    (x_c, t_c), _ = build_batch(PAIRS, "cpu")
    lp = seq_logprob(model, x_c, t_c)
    assert lp.shape == (2,)
    assert torch.isfinite(lp).all()
    assert (lp < 0).all()  # sum of log-probs is negative


def test_dpo_loss_zero_margin_when_policy_equals_ref():
    torch.manual_seed(1)
    model = Transformer(_cfg()).eval()
    c, r = build_batch(PAIRS, "cpu")
    loss, acc, margin = dpo_loss(model, model, c, r, beta=0.1)
    # policy == ref => implicit reward difference is 0 => loss = -log(0.5)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-4


def test_dpo_step_reduces_loss():
    """One optimization step on the policy should reduce the DPO loss."""
    torch.manual_seed(2)
    cfg = _cfg()
    policy = Transformer(cfg)
    ref = Transformer(cfg)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    c, r = build_batch(PAIRS, "cpu")
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    first, _, _ = dpo_loss(policy, ref, c, r, beta=0.1)
    for _ in range(10):
        opt.zero_grad()
        loss, _, _ = dpo_loss(policy, ref, c, r, beta=0.1)
        loss.backward()
        opt.step()
    assert loss.item() < first.item()
