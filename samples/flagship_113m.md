# Flagship 113M — generation samples

Model: 113M params (dim 768, 16 layers, GQA 12q/4kv, ctx 1024, own 16k BPE).
Training: bf16 + gradient checkpointing + torch.compile on one RTX 5070 Ti
(5.3 GB peak). A first run (655M tokens, val 1.257) was flagged as under-trained
by bits-per-byte; the longer run below (~984M tokens) reached **val 1.232,
perplexity 3.70, bits-per-byte 0.449**.

---

**Prompt:** `Once upon a time there was a little robot` — *longer run* (temp 0.8, seed 1)

> Once upon a time there was a little robot. He was very flexible, which means he was able to twist and turn in many different ways.
>
> One day the robot was walking through the forest, when he heard a voice. It was a little girl who said, "Hey robot, let me join you!" … "Come with me and I will show you something special!"
>
> So the robot followed the little girl. She showed him a secret path that lead into a dark cave. The robot was scared but he said, "Ok, let's go!" … Inside the chest were lots of colorful toys and books. The robot was so excited.

Note the step up from the first run: "flexible, which means he was able to
twist and turn", a multi-beat plot (meeting → cave → treasure), and sustained
dialogue.

---

## First-run samples (655M tokens)

Model: 113M params, 4000 steps (~655M tokens), val loss 1.257, perplexity 3.71.

Sampling: temperature 0.75–0.8, top-k 50. Reproduce with
`python -m llm.sample --ckpt checkpoints/flagship_113m/best.pt --prompt "..."`.

---

**Prompt:** `Once upon a time there was a little robot` (temp 0.8, seed 1)

> Once upon a time there was a little robot named Jake. Jake was very excited to explore the world. He had never seen so many wonders before!
>
> He went to the park and saw a pond full of ice and sparkling water. He had a wonderful time playing in it!
>
> Soon, the sun started to set and Jake had to go home. He was sad to leave, but he knew he'd be back soon.
>
> On the way home, Jake couldn't help but think about all the fun he had when he first met a new friend… He waved goodbye to the park and went back home. Jake was so excited to tell his family all about his day.

---

**Prompt:** `The old wizard looked at the map and said` (temp 0.75, seed 5)

> The old wizard looked at the map and said, "What do you see?" She pointed to the left path and the bald man smiled and said, "I see a place where people go to learn how to be kind to each other."
>
> The wizard decided to follow the map and find out what it was… He waved his wand and said, "The journey will be over soon."
>
> The old wizard smiled as he watched the little village go. He knew he had made the right choice in returning to the village and he was happy to have been kind.

---

### Notes

Compared to the 30M model, the flagship uses named characters (Jake), richer
vocabulary ("wonders", "sparkling"), longer coherent arcs, and the larger 16k
tokenizer + 1024 context. Note: the flagship's per-token val loss (1.257) is
**not** directly comparable to the 30M's (1.24) because they use different
tokenizers — a 16k vocab carries more information per token, so per-token
cross-entropy is naturally higher. Perplexity (3.71 vs 3.75) is likewise
tokenizer-dependent; bits-per-byte would be the fair cross-model metric.
