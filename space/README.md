---
title: HandmadeLLM
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# HandmadeLLM — a Llama-style LLM built from scratch

Playable demo of a decoder-only LLM (RoPE + RMSNorm + GQA + SwiGLU) whose
tokenizer, model and training loop were all written by hand — no Hugging Face
`Trainer`, no imported tokenizers. Type a TinyStories-style prompt and watch it
stream a story.

- Weights: [TyrkerS/handmadellm-30m](https://huggingface.co/TyrkerS/handmadellm-30m)
- Code + full writeup: https://github.com/TyrkerS/HandmadeLLM

Set the `HLLM_REPO` Space variable to `TyrkerS/handmadellm-113m` to serve the
larger flagship instead of the 30M.
