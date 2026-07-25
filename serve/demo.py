"""Playable Gradio demo — type a prompt, watch the story stream out.

Runs any checkpoint locally, and is also the entry point for a Hugging Face
Space (rename to app.py on the Space, or set the Space's app_file to this).

Run locally:
    pip install -r requirements-serve.txt
    HLLM_CKPT=checkpoints/tinystories_30m/best.pt python -m serve.demo
"""

from __future__ import annotations

import os

import gradio as gr
import torch
import torch.nn.functional as F

from llm.bpe import BPETokenizer
from llm.config import ModelConfig, _build
from llm.model import KVCache, Transformer

CKPT = os.environ.get("HLLM_CKPT", "checkpoints/tinystories_30m/best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
_model = Transformer(_build(ModelConfig, _ckpt["config"]["model"])).to(DEVICE).eval()
_model.load_state_dict(_ckpt["model"])
_tok = BPETokenizer.load(_ckpt["config"]["train"]["tokenizer_path"])


@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int, temperature: float, top_k: int):
    """Stream decoded text token-by-token, using the KV cache."""
    prompt = prompt or "Once upon a time"
    ids = torch.tensor([_tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    caches = [KVCache() for _ in _model.blocks]
    pos, cur = 0, ids
    acc: list[int] = []
    text = prompt
    for _ in range(int(max_new_tokens)):
        if pos + cur.shape[1] >= _model.cfg.max_seq_len:
            break
        logits, _ = _model(cur, caches=caches, start_pos=pos)
        pos += cur.shape[1]
        logits = logits[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(int(top_k), logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tid = int(nxt.item())
        if tid == _tok.eot_id:
            break
        acc.append(tid)
        text = prompt + " " + _tok.decode(acc)
        yield text
        cur = nxt


TITLE = "HandmadeLLM — a Llama-style LLM built from scratch"
DESC = (
    f"Serving `{os.path.basename(CKPT)}` on **{DEVICE}**. "
    "Tokenizer, model, training loop — all hand-written. "
    "Best with TinyStories-style prompts (e.g. *Once upon a time*)."
)

with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESC}")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Once upon a time there was a little robot", lines=2)
            max_new = gr.Slider(16, 400, value=200, step=8, label="Max new tokens")
            temp = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
            topk = gr.Slider(0, 200, value=50, step=1, label="Top-k")
            btn = gr.Button("Generate", variant="primary")
        out = gr.Textbox(label="Story", lines=16)
    btn.click(generate, [prompt, max_new, temp, topk], out)
    gr.Examples(
        [["Once upon a time there was a little robot", 200, 0.8, 50],
         ["The dragon was very sad because", 200, 0.7, 50],
         ["Lily and Tom found a key in the garden", 200, 0.8, 50]],
        [prompt, max_new, temp, topk],
    )

if __name__ == "__main__":
    # HLLM_SHARE=1 gives a temporary public URL (~72h) for live demos
    demo.launch(share=os.environ.get("HLLM_SHARE") == "1")
