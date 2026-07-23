"""Plot a training loss curve from a run's metrics.jsonl (no matplotlib needed —
emits an SVG directly, so the repo stays dependency-light).

Usage:
    python scripts/plot_loss.py checkpoints/tinystories_30m/metrics.jsonl --out samples/loss_30m.svg
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--out", default="samples/loss.svg")
    args = ap.parse_args()

    steps, losses = [], []
    for line in Path(args.metrics).read_text().splitlines():
        r = json.loads(line)
        steps.append(r["step"])
        losses.append(r["loss"])

    W, H, pad = 720, 360, 50
    x0, x1 = min(steps), max(steps)
    y0, y1 = 0.0, math.ceil(max(losses))

    def sx(s):
        return pad + (s - x0) / max(1, x1 - x0) * (W - 2 * pad)

    def sy(v):
        return H - pad - (v - y0) / (y1 - y0) * (H - 2 * pad)

    pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(steps, losses))
    grid = ""
    for gy in range(int(y1) + 1):
        yy = sy(gy)
        grid += f'<line x1="{pad}" y1="{yy:.1f}" x2="{W - pad}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        grid += f'<text x="{pad - 8}" y="{yy + 4:.1f}" font-size="12" fill="#6b7280" text-anchor="end">{gy}</text>'

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
{grid}
<polyline points="{pts}" fill="none" stroke="#2563eb" stroke-width="2"/>
<text x="{W/2}" y="24" font-size="15" fill="#111827" text-anchor="middle" font-weight="600">TinyStories 30M — training loss</text>
<text x="{W/2}" y="{H-12}" font-size="12" fill="#6b7280" text-anchor="middle">step (0 to {x1:,}) — final {losses[-1]:.3f}</text>
</svg>'''
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8")
    print(f"wrote {args.out} ({len(steps)} points, final loss {losses[-1]:.3f})")


if __name__ == "__main__":
    main()
