# TinyStories 30M — generation samples

Model: 28M params (dim 512, 8 layers, GQA 8q/4kv, ctx 512, own 8k BPE).
Training: 12,000 steps, effective batch 131k tokens, final val loss 1.24, held-out perplexity 3.75.
Sampling: temperature 0.7–0.8, top-k 50. Reproduce with `python -m llm.sample --ckpt checkpoints/tinystories_30m/best.pt --prompt "..."`.

---

**Prompt:** `Once upon a time there was a little robot` (temp 0.8, seed 1)

> Once upon a time there was a little robot who was very excited. He was always looking for new things to add to his collection of friends. He had all of his robot friends that he loved.
>
> One day, he went out for a walk. As he was walking, he saw a small box. It was a very big box, and his robot friends were very excited. They all wanted to add something to the box.
>
> The robot thought for a minute and then said, "Let's add something to the box!" He brought out a toy truck, and they all smiled.
>
> The robot asked, "What can we add to the truck?" The other robots said, "We can add some books!" So they all added books to the box and made it look nice.
>
> When they were finished, the new toy truck was bright and shiny. The robot was so excited. He played with it

---

**Prompt:** `Lily and Tom found a key in the garden` (temp 0.8, seed 3)

> Lily and Tom found a key in the garden. They were happy. They opened the box and found many toys inside. There were dolls, cars, books, and puzzles. Lily and Tom wanted to play with everything.
>
> "Look, a doll!" Lily said. She picked up the doll and gave it a hug.
>
> "I want the car!" Tom said. He grabbed the car from Lily and made it zoom on the floor.
>
> "Mine!" Lily said. She pulled the doll from Tom and hugged it tight.
>
> "No, mine!" Tom said. He grabbed the doll from Lily and pulled hard.
>
> Lily and Tom pulled and pulled. They both wanted the doll very much. They did not want to share.
>
> The doll started to cry. It did not like to be pulled

---

**Prompt:** `The dragon was very sad because` (temp 0.7, seed 7)

> The dragon was very sad because he had no friends. He thought about all the other animals in the forest and wished he had someone to play with. One day, he had an idea. He went to the village and asked the villagers if they wanted to be his friends.
>
> The villagers were very surprised, but they agreed to let the dragon be. They gave him some food and water and he became very happy. He played with the children and ate the food and drank the water. He even learned how to bow to the villagers when they visited.
>
> But one day, the dragon got very sick and had to go away. The villagers were very sad and didn't know what to do. They missed him very much and wished he was still there to play with them.
>
> The moral of the story

---

### Notes

- Grammar is clean, stories have a beginning/middle/end, dialogue is correctly attributed, and the model has picked up TinyStories' "moral of the story" convention.
- Failure modes (honest): occasional logical slips ("a small box. It was a very big box"), and the model sometimes trails mid-sentence at the token budget. Expected at 28M params — the point is coherence, not a production assistant.
