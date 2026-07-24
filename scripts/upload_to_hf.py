"""Publish a trained checkpoint to the Hugging Face Hub with a generated model card.

Weights are intentionally gitignored (checkpoints are ~300 MB / ~1.3 GB); the
Hub is where they belong. Run this yourself once you're logged in:

    pip install huggingface_hub
    huggingface-cli login                      # paste your write token
    python scripts/upload_to_hf.py --ckpt checkpoints/tinystories_30m/best.pt \
        --repo <your-username>/handmadellm-30m --tokenizer data/tinystories/tokenizer.json

The script uploads the checkpoint + tokenizer + a README.md model card. It does
NOT log in for you and never handles your token.
"""

from __future__ import annotations

import argparse
from pathlib import Path

CARD = """---
license: mit
tags: [handmade, from-scratch, tinystories, llama, gqa, rope]
library_name: pytorch
---

# {repo}

A **Llama-style decoder-only LLM built entirely from scratch** — own byte-level
BPE tokenizer, RoPE + RMSNorm + grouped-query attention + SwiGLU, hand-written
training loop. Trained on TinyStories on a single RTX 5070 Ti.

- **Params:** {params}
- **Architecture:** dim {dim}, {layers} layers, {heads} heads ({kv} KV heads, GQA), context {ctx}, vocab {vocab}
- **Val bits-per-byte:** {bpb}

Code, training scripts, evals, quantization, a fused Triton kernel, and the full
writeup: **https://github.com/<your-username>/HandmadeLLM**

## Usage

```python
import torch
from llm.config import ModelConfig, _build
from llm.model import Transformer
from llm.bpe import BPETokenizer

ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model = Transformer(_build(ModelConfig, ckpt["config"]["model"]))
model.load_state_dict(ckpt["model"]); model.eval()
tok = BPETokenizer.load("tokenizer.json")

ids = torch.tensor([tok.encode("Once upon a time")])
out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=50, eos_id=tok.eot_id)
print(tok.decode(out[0].tolist()))
```

This model is a research/portfolio artifact trained on TinyStories; it writes
simple children's-story English and is not an instruction-following assistant
(unless you loaded the SFT/DPO variant).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="<username>/<repo-name>")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--bpb", default="see repo")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, create_repo

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = ckpt["config"]["model"]

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from llm.config import ModelConfig, _build
    from llm.model import Transformer
    n = Transformer(_build(ModelConfig, m)).num_params()

    card = CARD.format(
        repo=args.repo, params=f"{n/1e6:.0f}M", dim=m["dim"], layers=m["n_layers"],
        heads=m["n_heads"], kv=m["n_kv_heads"], ctx=m["max_seq_len"],
        vocab=m["vocab_size"], bpb=args.bpb,
    )

    create_repo(args.repo, exist_ok=True, private=args.private)
    api = HfApi()
    api.upload_file(path_or_fileobj=args.ckpt, path_in_repo="best.pt", repo_id=args.repo)
    api.upload_file(path_or_fileobj=args.tokenizer, path_in_repo="tokenizer.json", repo_id=args.repo)
    tmp = Path("_MODELCARD.md")
    tmp.write_text(card, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="README.md", repo_id=args.repo)
    tmp.unlink()
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
