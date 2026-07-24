# DPO — preference alignment (Phase 5 stretch)

Direct Preference Optimization from scratch ([llm/dpo.py](../llm/dpo.py)),
starting from the SFT checkpoint. Preference pairs: **chosen** = the real
TinyStories-Instruct story, **rejected** = a generation from the pre-SFT base
model for the same prompt. Frozen reference = the SFT model. β = 0.1, 1 epoch,
2,400 train / 100 val pairs.

## Result

Held-out **preference accuracy** — the fraction of pairs where the model assigns
higher total log-probability to the gold story than to the base-model ramble
(a clean, model-comparable metric):

| model | preference accuracy |
|---|---|
| SFT (reference) | 0.700 |
| **DPO (aligned)** | **0.970** |

DPO moved the model from preferring the gold response 70% of the time to 97% —
a clear alignment gain, and generation quality held up (the aligned model even
introduces more narrative tension, e.g. a real conflict in the "learn to share"
story).

## Honest caveat — over-optimization

During training the implicit-reward **margins exploded** (log-prob differences
in the hundreds) and the training preference accuracy hit 1.0 almost
immediately. That's the classic DPO over-optimization signal: with a small β and
enough steps the policy drifts far from the reference. Here the outputs stayed
coherent (SFT anchored them well), but in a longer/production run I would temper
this with a **higher β (0.3–0.5), early stopping on val preference accuracy, and
fewer steps** rather than driving the margin to the moon. Reporting it because
pretending the margins were tame would be dishonest — the *held-out* metric and
sample quality are what actually validate the run.

Reproduce:
```
python scripts/prepare_dpo.py --base checkpoints/tinystories_30m/best.pt --sft-data data/sft/train.pt
python -m llm.dpo --init checkpoints/sft_30m/best.pt --data data/dpo/train.pt --val data/dpo/val.pt --out checkpoints/dpo_30m
```
