| Mode | Model | Retriever | F1 [95% CI] | Precision | Recall | Gold cited / shown, not cited / never shown | $ per 1k questions | Median latency | Searches |
|---|---|---|---|---|---|---|---|---|---|
| closed book | Qwen3 30B-A3B | — | **0.4** [0.0, 0.9] | 0.5 | 0.6 | 1 / 0 / 99 | $0.20 | 3.4 s | 0.0 |
| closed book | Claude Haiku 5.5 | — | **18.1** [14.5, 21.9] | 26.0 | 19.7 | 20 / 0 / 80 | $0.64 | 6.6 s | 0.0 |
| one shot (top 20) | Qwen3 30B-A3B | BM25 | **28.5** [24.2, 32.8] | 33.2 | 33.9 | 34 / 9 / 57 | $0.83 | 5.4 s | 0.0 |
| one shot (top 20) | Qwen3 30B-A3B | bge-m3 | **31.3** [27.2, 35.6] | 36.5 | 37.9 | 38 / 12 / 50 | $0.70 | 6.0 s | 0.0 |
| one shot (top 20) | Qwen3 30B-A3B | bge-m3 fine-tuned | **39.8** [35.7, 44.1] | 46.4 | 48.5 | 49 / 26 / 25 | $0.70 | 6.4 s | 0.0 |
| one shot (top 20) | Claude Haiku 5.5 | BM25 | **32.1** [27.4, 36.9] | 38.3 | 35.9 | 36 / 7 / 57 | $1.50 | 4.2 s | 0.0 |
| one shot (top 20) | Claude Haiku 5.5 | bge-m3 | **36.6** [32.3, 41.1] | 43.2 | 43.2 | 43 / 7 / 50 | $1.34 | 4.4 s | 0.0 |
| one shot (top 20) | Claude Haiku 5.5 | bge-m3 fine-tuned | **42.5** [38.4, 46.6] | 50.4 | 51.5 | 51 / 23 / 25 | $1.34 | 4.5 s | 0.0 |
| one shot (top 20) | Claude Sonnet 5.5 | bge-m3 fine-tuned | **42.0** [37.7, 46.3] | 48.4 | 50.5 | 50 / 24 / 25 | $22.63 | 2.8 s | 0.0 |
| agent | Qwen3 30B-A3B | BM25 | **26.0** [22.2, 29.9] | 29.8 | 32.1 | 32 / 9 / 59 | $2.22 | 25.2 s | 2.9 |
| agent | Qwen3 30B-A3B | bge-m3 | **27.9** [24.1, 31.8] | 30.6 | 35.3 | 35 / 12 / 53 | $1.93 | 30.4 s | 2.5 |
| agent | Qwen3 30B-A3B | bge-m3 fine-tuned | **31.1** [27.1, 35.1] | 36.0 | 39.3 | 39 / 27 / 34 | $1.90 | 32.2 s | 2.6 |
| agent | Claude Haiku 5.5 | BM25 | **44.3** [39.7, 48.9] | 54.5 | 48.0 | 48 / 21 / 31 | $7.24 | 11.5 s | 6.2 |
| agent | Claude Haiku 5.5 | bge-m3 | **43.3** [38.7, 47.9] | 52.4 | 48.6 | 49 / 20 / 31 | $6.82 | 10.9 s | 5.7 |
| agent | Claude Haiku 5.5 | bge-m3 fine-tuned | **44.9** [40.4, 49.4] | 54.0 | 49.4 | 49 / 34 / 16 | $6.90 | 12.4 s | 6.1 |

| Change | ΔF1 [95% CI] |
|---|---|
| Qwen3 30B-A3B, BM25: one shot → agent | -2.4 [-5.7, +0.9] |
| Qwen3 30B-A3B, bge-m3: one shot → agent | -3.4 [-6.3, -0.4] |
| Qwen3 30B-A3B, bge-m3 fine-tuned: one shot → agent | -8.7 [-12.2, -5.3] |
| Qwen3 30B-A3B, one shot (top 20): bge-m3 → fine-tuned | +8.5 [+4.9, +12.1] |
| Qwen3 30B-A3B, agent: bge-m3 → fine-tuned | +3.2 [-0.2, +6.7] |
| Claude Haiku 5.5, BM25: one shot → agent | +12.2 [+7.9, +16.5] |
| Claude Haiku 5.5, bge-m3: one shot → agent | +6.7 [+2.7, +10.7] |
| Claude Haiku 5.5, bge-m3 fine-tuned: one shot → agent | +2.4 [-1.3, +6.2] |
| Claude Haiku 5.5, one shot (top 20): bge-m3 → fine-tuned | +5.9 [+1.7, +10.1] |
| Claude Haiku 5.5, agent: bge-m3 → fine-tuned | +1.6 [-1.2, +4.5] |
| One shot, bge-m3 fine-tuned: Haiku → Sonnet | -0.5 [-3.1, +2.1] |
