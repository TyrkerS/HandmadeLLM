"""Phase 4: FastAPI serving with token streaming.

Run:
    pip install fastapi uvicorn
    HLLM_CKPT=checkpoints/tinystories_30m/best.pt uvicorn serve.app:app --port 8000

Endpoints:
    GET  /health
    POST /generate            -> {"text": ...}          (blocking)
    POST /generate/stream     -> text/event-stream       (SSE token stream)

Request body: {"prompt": str, "max_new_tokens": int, "temperature": float, "top_k": int}
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm.bpe import BPETokenizer
from llm.config import ModelConfig, _build
from llm.model import KVCache, Transformer

CKPT = os.environ.get("HLLM_CKPT", "checkpoints/tinystories_30m/best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="HandmadeLLM")
_model: Transformer | None = None
_tok: BPETokenizer | None = None


def _load() -> None:
    global _model, _tok
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    mcfg = _build(ModelConfig, ckpt["config"]["model"])
    _model = Transformer(mcfg).to(DEVICE)
    _model.load_state_dict(ckpt["model"])
    _model.eval()
    _tok = BPETokenizer.load(ckpt["config"]["train"]["tokenizer_path"])


@app.on_event("startup")
def startup() -> None:
    _load()


class GenRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 50


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE, "ckpt": CKPT, "loaded": _model is not None}


@torch.inference_mode()
def _stream_tokens(req: GenRequest):
    """Generator yielding decoded text incrementally, using the KV cache."""
    ids = torch.tensor([_tok.encode(req.prompt)], dtype=torch.long, device=DEVICE)
    caches = [KVCache() for _ in _model.blocks]
    pos = 0
    cur = ids
    prev_text = ""
    acc: list[int] = []
    for _ in range(req.max_new_tokens):
        if pos + cur.shape[1] >= _model.cfg.max_seq_len:
            break
        logits, _ = _model(cur, caches=caches, start_pos=pos)
        pos += cur.shape[1]
        logits = logits[:, -1, :]
        if req.temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / req.temperature
            if req.top_k:
                kth = torch.topk(logits, min(req.top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        tid = int(nxt.item())
        if tid == _tok.eot_id:
            break
        acc.append(tid)
        # decode incrementally; emit only the newly completed suffix (handles multi-byte tokens)
        text = _tok.decode(acc)
        if "�" not in text[len(prev_text):]:
            yield text[len(prev_text):]
            prev_text = text
        cur = nxt


@app.post("/generate")
def generate(req: GenRequest) -> dict:
    return {"text": "".join(_stream_tokens(req))}


@app.post("/generate/stream")
def generate_stream(req: GenRequest) -> StreamingResponse:
    def event_gen():
        for piece in _stream_tokens(req):
            yield f"data: {json.dumps({'token': piece})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
