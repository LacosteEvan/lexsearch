"""Fine-tune an embedding model on BSARD's training questions.

One example per (question, answering article), plus one hard negative: an article the base model ranks high that the
lawyers did not label. Negatives are drawn from ranks 10 to 60, not the very top: BSARD's labels are incomplete, and the
top unlabeled articles are often relevant too. The loss is in-batch contrastive (MultipleNegativesRankingLoss): every
other article in the batch is a negative as well, and batches never hold the same question or article twice.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import data
from .retrievers import PREFIXES, Dense, _device


def triplets(base: Dense, split: str = "fit", synthetic: int = 0, seed: int = 0) -> list[dict]:
    articles = data.articles()
    qs = list(data.questions(split))
    rng = random.Random(seed)
    if synthetic:
        qs += rng.sample(data.questions("synthetic"), synthetic)
    ranked = base.search([q.text for q in qs], k=60)
    q_prefix, p_prefix = PREFIXES.get(base.model_id, ("", ""))
    rows = []
    for q, run in zip(qs, ranked, strict=True):
        pool = [a for a in run[10:60] if a not in q.gold] or [a for a in run if a not in q.gold]
        for a in sorted(q.gold):
            rows.append({"anchor": q_prefix + q.text, "positive": p_prefix + articles[a].as_passage(),
                         "negative": p_prefix + articles[rng.choice(pool)].as_passage()})
    rng.shuffle(rows)
    return rows


def train(model_id: str, out: Path, synthetic: int = 0, epochs: int = 3, batch_size: int = 32, lr: float = 2e-5,
          max_seq_length: int = 384, seed: int = 0) -> Path:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    base = Dense(model_id, list(data.articles().values()))
    rows = triplets(base, synthetic=synthetic, seed=seed)
    print(f"{len(rows)} training triplets", flush=True)
    del base
    device = _device()
    # Eager attention on Apple GPUs: PyTorch's fused attention there does not support dropout, which training uses.
    kwargs = {"attn_implementation": "eager"} if device == "mps" else {}
    model = SentenceTransformer(model_id, device=device, model_kwargs=kwargs)
    model.max_seq_length = max_seq_length
    # Cached MNRL: the loss of a large batch, computed in mini-batches that fit in memory.
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8 if device == "mps" else 32)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES, logging_steps=10,
        save_strategy="no", seed=seed, report_to="none", dataloader_drop_last=True, bf16=device == "cuda")
    SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), loss=loss).train()
    model.max_seq_length = 512
    model.save(str(out))
    (out / "lexsearch.json").write_text(json.dumps({"base": model_id, "synthetic": synthetic, "epochs": epochs,
                                                    "batch_size": batch_size, "lr": lr, "triplets": len(rows)}))
    return out
