# Instruction tuning (SFT) — before vs after

Base model = pretrained 30M on plain TinyStories. SFT = fine-tuned on 30k
TinyStories-Instruct (prompt, response) pairs with prompt-masked loss (3 epochs,
lr 2e-5). Same prompts, temperature 0.7, top-k 50. Reproduce:
`python scripts/sft_compare.py --ckpt <checkpoint> --label <before|after>`.

The instruction format is:
```
Features: <e.g. Dialogue>
Words: <words the story must use>
Summary: <one-line plot the story must follow>
Story:
```

---

## Prompt 1 — Words: dog, jump, happy

**BEFORE (base model):** rambles, ignores the required words (talks about a girl
and "Mint", not a dog jumping), and emits spurious `Story:`/`Apparent:` tokens:

> That sounds like a fun game. The boy and the girl laughed and smiled. They had a wonderful time playing Mint in the backyard.

**AFTER (SFT):**

<!-- SFT_AFTER_1 -->

---

## Prompt 2 — Words: share, cookie, friend / MoralValue

**BEFORE:** collapses to a single moral line, no story:

> Apparent: We should always share the things we have.

**AFTER (SFT):**

<!-- SFT_AFTER_2 -->

---

## Prompt 3 — Words: boat, river, brave

**BEFORE:** loops, injects literal `Story:` markers and a character literally named
"Story":

> "I want to sail the boat!" … The little girl helped Story onto the boat. They sailed down the river … Story waved goodbye…

**AFTER (SFT):**

<!-- SFT_AFTER_3 -->
