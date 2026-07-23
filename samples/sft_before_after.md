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

**AFTER (SFT):** uses **all** required words (happy, dog, jump, yard), follows the summary, clean story structure:

> Once upon a time, there was a happy dog. The dog loved to play and jump all day. One day, the dog saw a big ball… The dog jumped and played with the ball all day long. They had so much fun together.

---

## Prompt 2 — Words: share, cookie, friend / MoralValue

**BEFORE:** collapses to a single moral line, no story:

> Apparent: We should always share the things we have.

**AFTER (SFT):** a real story about two friends sharing a cookie — words + summary honored:

> Once upon a time, there were two friends who liked to share… The girl had a cookie and the boy had a cookie. They decided to share the cookie… The friends laughed and smiled as they shared the cookies. They had a wonderful time.

---

## Prompt 3 — Words: boat, river, brave

**BEFORE:** loops, injects literal `Story:` markers and a character literally named
"Story":

> "I want to sail the boat!" … The little girl helped Story onto the boat. They sailed down the river … Story waved goodbye…

**AFTER (SFT):** brave girl sails a boat down the river — words + summary honored, no spurious markers:

> Once upon a time, there was a little girl. She was very brave and always wanted to explore. One day, the little girl decided to sail down the river. She got on a boat and sailed away… The little girl was very proud of her brave sailing adventure.

---

### Takeaway

The base model (plain TinyStories) treated the instruction fields as noise — it rambled, ignored the required words, and leaked literal `Story:`/`Apparent:` tokens. After SFT with **prompt-masked loss** on 30k instruction pairs, it reliably follows the format: incorporates every required word, respects the one-line summary, and produces a clean, self-contained story. SFT val loss reached **1.14**. That's the difference between "generates text" and "follows instructions."
