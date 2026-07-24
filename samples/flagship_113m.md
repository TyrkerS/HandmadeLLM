# Flagship 113M — generation samples

Model: 113M params (dim 768, 16 layers, GQA 12q/4kv, ctx 1024, own 16k BPE).
Training: 4000 steps, effective batch 164k tokens (~655M tokens total), bf16 +
gradient checkpointing + torch.compile, ~5 h on one RTX 5070 Ti (5.3 GB peak).
Final val loss 1.257, held-out perplexity 3.71 (16k vocab).

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
