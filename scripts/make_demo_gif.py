"""Render a terminal-style GIF of the model writing a story token-by-token.

Uses REAL generation from a checkpoint (no faking) and animates the text
appearing, so the README shows what the model actually does.

Usage:
    python scripts/make_demo_gif.py --ckpt checkpoints/tinystories_30m/best.pt \
        --out samples/demo.gif
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.bpe import BPETokenizer  # noqa: E402
from llm.config import ModelConfig, _build  # noqa: E402
from llm.model import KVCache, Transformer  # noqa: E402

# terminal palette
BG = (13, 17, 23)
FG = (201, 209, 217)
PROMPT_COL = (88, 166, 255)
ACCENT = (63, 185, 80)
W, H = 900, 520
PAD = 28
LINE_H = 26
WRAP = 78


def load_font(size):
    for name in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(prompt: str, body: str, font, cursor: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # header bar: clean, neutral (no OS window chrome)
    d.rectangle([0, 0, W, 40], fill=(22, 27, 34))
    d.line([0, 40, W, 40], fill=(48, 54, 61), width=1)
    d.text((PAD, 12), "HandmadeLLM", font=font, fill=ACCENT)
    d.text((PAD + int(d.textlength("HandmadeLLM", font=font)) + 14, 12),
           "— generating", font=font, fill=(139, 148, 158))

    y = 58
    d.text((PAD, y), "$ ", font=font, fill=ACCENT)
    d.text((PAD + 22, y), prompt, font=font, fill=PROMPT_COL)
    y += LINE_H + 8

    lines = []
    for para in body.split("\n"):
        lines.extend(textwrap.wrap(para, WRAP) or [""])
    for line in lines[-15:]:
        d.text((PAD, y), line, font=font, fill=FG)
        y += LINE_H
    if cursor and lines:
        last = lines[-1] if len(lines) <= 15 else lines[14]
        cx = PAD + int(d.textlength(last, font=font))
        cy = min(y - LINE_H, H - LINE_H)
        d.rectangle([cx + 2, cy, cx + 12, cy + 18], fill=ACCENT)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tinystories_30m/best.pt")
    ap.add_argument("--out", default="samples/demo.gif")
    ap.add_argument("--prompt", default="Once upon a time there was a little robot")
    ap.add_argument("--max-new", type=int, default=110)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = Transformer(_build(ModelConfig, ckpt["config"]["model"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = BPETokenizer.load(ckpt["config"]["train"]["tokenizer_path"])
    font = load_font(19)

    # real token-by-token generation, capturing a frame every few tokens
    ids = torch.tensor([tok.encode(args.prompt)], device=device)
    caches = [KVCache() for _ in model.blocks]
    pos, cur, acc = 0, ids, []
    frames = [render_frame(args.prompt, "", font, True)]
    with torch.inference_mode():
        for step in range(args.max_new):
            logits, _ = model(cur, caches=caches, start_pos=pos)
            pos += cur.shape[1]
            logits = logits[:, -1, :] / 0.8
            kth = torch.topk(logits, 50).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            tid = int(nxt.item())
            if tid == tok.eot_id:
                break
            acc.append(tid)
            cur = nxt
            if step % 2 == 0:
                frames.append(render_frame(args.prompt, tok.decode(acc), font, True))
    # hold the final frame
    final = render_frame(args.prompt, tok.decode(acc), font, False)
    frames.extend([final] * 12)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=110, loop=0, optimize=True)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
