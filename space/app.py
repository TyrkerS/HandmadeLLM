"""Hugging Face Space entry point — playable HandmadeLLM demo.

Downloads the model + tokenizer from the Hub repo and streams generated stories.
Self-contained: vendors the minimal model/tokenizer code so the Space needs only
torch + gradio + huggingface_hub (no local package install).

To deploy: create a Gradio Space, upload this file, requirements.txt and
README.md. Set REPO_ID below to your model repo.
"""

from __future__ import annotations

import os

import gradio as gr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# import the from-scratch model + tokenizer straight from the uploaded package
# (the Space also vendors llm/ — see requirements/README notes)
from llm.bpe import BPETokenizer
from llm.config import ModelConfig, _build
from llm.model import KVCache, Transformer

REPO_ID = os.environ.get("HLLM_REPO", "TyrkerS/handmadellm-30m")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = hf_hub_download(REPO_ID, "best.pt")
tok_path = hf_hub_download(REPO_ID, "tokenizer.json")

_ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
_model = Transformer(_build(ModelConfig, _ckpt["config"]["model"])).to(DEVICE).eval()
_model.load_state_dict(_ckpt["model"])
_tok = BPETokenizer.load(tok_path)


@torch.inference_mode()
def generate(prompt, max_new_tokens, temperature, top_k):
    prompt = prompt or "Once upon a time"
    ids = torch.tensor([_tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    caches = [KVCache() for _ in _model.blocks]
    pos, cur, acc = 0, ids, []
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
        yield prompt + " " + _tok.decode(acc)
        cur = nxt


with gr.Blocks(title="HandmadeLLM") as demo:
    gr.Markdown(
        "# HandmadeLLM — a Llama-style LLM built from scratch\n"
        f"Serving `{REPO_ID}` on **{DEVICE}**. Tokenizer, model, training loop — all "
        "hand-written. Best with TinyStories-style prompts (e.g. *Once upon a time*).\n"
        "Code: https://github.com/TyrkerS/HandmadeLLM"
    )
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
    demo.launch()
