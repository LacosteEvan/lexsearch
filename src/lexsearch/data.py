"""BSARD: 22,633 Belgian statutory articles in French, and questions asked by citizens, each labeled by lawyers with
the articles that answer it (Louis & Spanakis, ACL 2022). CC BY-NC-SA 4.0: downloaded from the Hugging Face Hub at run
time, never copied into this repository.

Splits used here:
- train (886 questions) is cut in two by a hash of the question id: `fit` (~80%) to fine-tune, `dev` (~20%) to choose
  between methods. Every design decision is made on dev.
- test (222 questions) is only read for the final tables.
- synthetic (113,165 machine-written questions, one article each) is optional extra training data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache

HF_NAME = "maastrichtlawtech/bsard"
DEV_SHARE = 0.2


@dataclass(frozen=True)
class Article:
    id: str
    reference: str  # "Art. 1.1.1, Code Bruxellois de l'Air, ..."
    path: str  # headings above the article: book, chapter, section...
    text: str

    def as_passage(self) -> str:
        """What every retriever indexes: the citation, the headings, then the text."""
        return f"{self.reference}\n{self.path}\n{self.text}" if self.path else f"{self.reference}\n{self.text}"


@dataclass(frozen=True)
class Question:
    id: str
    text: str  # the question, plus the context the citizen gave ("Pendant la grossesse")
    category: str
    gold: frozenset[str]  # ids of the articles that answer it


def _hash_share(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


@cache
def articles() -> dict[str, Article]:
    from datasets import load_dataset

    rows = load_dataset(HF_NAME, "corpus")["corpus"]
    return {str(r["id"]): Article(str(r["id"]), r["reference"], r["description"] or "", r["article"]) for r in rows}


def _question(r: dict) -> Question:
    text = r["question"].strip()
    if r.get("extra_description"):
        text = f"{text} (Contexte : {r['extra_description'].strip()})"
    return Question(str(r["id"]), text, r.get("category") or "", frozenset(r["article_ids"].split(",")))


@cache
def questions(split: str) -> list[Question]:
    """split: fit, dev, test or synthetic."""
    from datasets import load_dataset

    hf = load_dataset(HF_NAME, "questions")
    if split in ("fit", "dev"):
        rows = [_question(r) for r in hf["train"]]
        return [q for q in rows if (_hash_share(q.id) < DEV_SHARE) == (split == "dev")]
    if split in ("test", "synthetic"):
        return [_question(r) for r in hf[split]]
    raise ValueError(f"unknown split {split!r}")


def seen_articles() -> frozenset[str]:
    """Articles that answer at least one question the retriever is fine-tuned on. It can learn these by heart, so the
    results are also reported on the questions whose answers lie outside this set."""
    return frozenset(a for q in questions("fit") for a in q.gold)
