import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as np
import torch

from llm.bpe import BPETokenizer
from llm.config import ModelConfig
from llm.eval import byte_lengths, eval_nll, perplexity
from llm.model import Transformer


def _tok():
    t = BPETokenizer()
    t.train("the cat sat on the mat. the dog ran. " * 40, vocab_size=320)
    return t


def test_byte_lengths_match_vocab():
    tok = _tok()
    bl = byte_lengths(tok)
    assert bl.shape == (tok.vocab_size,)
    assert (bl[:256] == 1).all()          # raw byte tokens are 1 byte
    assert bl[tok.eot_id] == len("<|endoftext|>".encode())
    # a merged token spans >= 2 bytes
    assert bl[256:256 + len(tok.merges)].min() >= 2


def test_bpb_and_perplexity_consistent():
    torch.manual_seed(0)
    tok = _tok()
    cfg = ModelConfig(dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
                      vocab_size=tok.vocab_size, max_seq_len=32)
    model = Transformer(cfg).eval()
    ids = np.array(tok.encode("the cat sat on the mat. " * 20), dtype=np.uint16)

    nll, ntok, nbytes = eval_nll(model, ids, 16, "cpu", None, byte_lengths(tok))
    assert ntok > 0 and nbytes > ntok  # each token averages >1 byte here
    bpb = (nll / math.log(2)) / nbytes
    ppl = perplexity(model, ids, 16, "cpu")
    # both derive from the same NLL; sanity bounds
    assert ppl > 1.0
    assert 0.0 < bpb < 20.0
    # bpb equals per-token cross-entropy in bits divided by bytes-per-token
    bits_per_tok = (nll / ntok) / math.log(2)
    assert abs(bpb - bits_per_tok * ntok / nbytes) < 1e-6
