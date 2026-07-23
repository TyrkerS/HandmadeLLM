"""Generate text from a trained checkpoint.

Usage:
    python -m llm.sample --ckpt checkpoints/run/best.pt --prompt "Once upon a time" --max-new 200
"""

from __future__ import annotations

import argparse

import torch

from .bpe import BPETokenizer
from .config import Config, _build, ModelConfig
from .model import Transformer


def load_model(ckpt_path: str, device: str) -> tuple[Transformer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=None, help="defaults to tokenizer_path from the ckpt config")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model, ckpt = load_model(args.ckpt, device)
    tok_path = args.tokenizer or ckpt["config"]["train"]["tokenizer_path"]
    tok = BPETokenizer.load(tok_path)

    ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        eos_id=tok.eot_id,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
