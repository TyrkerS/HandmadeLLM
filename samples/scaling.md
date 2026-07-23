# Scaling study (fixed ~110M-token budget, TinyStories)

| model | dim×layers | non-emb params | final val loss |
|---|---|---|---|
| s1_tiny | 256×4 | 3.0M | 1.998 |
| s2_small | 384×6 | 9.7M | 1.739 |
| s3_med | 512×8 | 23.6M | 1.615 |
| s4_large | 640×10 | 45.5M | 1.540 |

Power-law fit: **L ≈ 2.197 · N^(−0.096)**, R² = 0.987 (data-controlled, fixed token budget).
