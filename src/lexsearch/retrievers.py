"""Retrievers over the 22,633 articles. Each one maps questions to ranked article ids.

- BM25: keywords, with French stemming and stopwords.
- Dense: a sentence-embedding model, cosine similarity; article embeddings are cached in cache/emb/.
- Hybrid: reciprocal rank fusion of two rankings.
- Rerank: a cross-encoder rescores the top candidates of another retriever.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import Article

CACHE = Path("cache")

# Query and passage prefixes each model was trained with; a model missing here gets none.
PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}
QUERY_PROMPTS = {  # models that take a named prompt for queries instead of a prefix
    "Qwen/Qwen3-Embedding-0.6B": "query",
}


def base_model(model: str) -> str:
    """The Hub model a fine-tuned checkpoint started from (its prefixes and prompts carry over)."""
    meta = Path(model) / "lexsearch.json"
    return json.loads(meta.read_text())["base"] if meta.exists() else model


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class BM25:
    name = "bm25"

    def __init__(self, articles: list[Article]) -> None:
        import bm25s
        import Stemmer

        self.ids = [a.id for a in articles]
        self.stemmer = Stemmer.Stemmer("french")
        self.index = bm25s.BM25()
        self.index.index(self._tokenize([a.as_passage() for a in articles]), show_progress=False)

    def _tokenize(self, texts: list[str]):
        import bm25s

        return bm25s.tokenize(texts, stopwords="fr", stemmer=self.stemmer.stemWords, show_progress=False)

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        docs, _ = self.index.retrieve(self._tokenize(queries), k=k, show_progress=False)
        return [[self.ids[i] for i in row] for row in docs]


class Dense:
    def __init__(self, model: str, articles: list[Article], max_seq_length: int = 512, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = Path(model).name if Path(model).exists() else model
        self.model_id = model
        self.encoder = SentenceTransformer(model, device=_device(), trust_remote_code=False)
        self.encoder.max_seq_length = max_seq_length
        self.query_prefix, self.passage_prefix = PREFIXES.get(base_model(model), ("", ""))
        self.ids = [a.id for a in articles]
        self.batch_size = batch_size
        self.matrix = self._passages([a.as_passage() for a in articles])

    def _passages(self, texts: list[str]) -> np.ndarray:
        key = hashlib.sha256(("\n\x00".join(texts) + self.passage_prefix).encode()).hexdigest()[:12]
        path = CACHE / "emb" / f"{self.model_id.replace('/', '__')}-{self.encoder.max_seq_length}-{key}.npy"
        if path.exists():
            return np.load(path)
        # Longest first: batches of similar length waste less padding.
        order = np.argsort([-len(t) for t in texts])
        vecs = self.encoder.encode([self.passage_prefix + texts[i] for i in order], batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        matrix = np.empty_like(vecs)
        matrix[order] = vecs
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix.astype(np.float32))
        return matrix

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        prompt = QUERY_PROMPTS.get(base_model(self.model_id))
        kw = {"prompt_name": prompt} if prompt else {}
        return self.encoder.encode([self.query_prefix + q for q in queries], batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, **kw)

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        scores = self.embed_queries(queries) @ self.matrix.T
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        out = []
        for row, cand in zip(scores, top, strict=True):
            out.append([self.ids[i] for i in cand[np.argsort(-row[cand])]])
        return out


def rrf(rankings: list[list[str]], k: int = 60, depth: int = 100) -> list[str]:
    """Reciprocal rank fusion: each list votes 1 / (k + rank) for its articles."""
    score: dict[str, float] = {}
    for ranked in rankings:
        for rank, a in enumerate(ranked):
            score[a] = score.get(a, 0.0) + 1 / (k + rank + 1)
    return sorted(score, key=score.__getitem__, reverse=True)[:depth]


class Hybrid:
    def __init__(self, *retrievers) -> None:
        self.retrievers = retrievers
        self.name = "+".join(r.name for r in retrievers)

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        runs = [r.search(queries, k) for r in self.retrievers]
        return [rrf([run[i] for run in runs], depth=k) for i in range(len(queries))]


class Rerank:
    """Rescore the top `depth` candidates of `first` with a cross-encoder; the rest keep their order behind."""

    def __init__(self, first, model: str, articles: dict[str, Article], depth: int = 50, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder

        self.first, self.depth, self.articles = first, depth, articles
        self.encoder = CrossEncoder(model, device=_device(), max_length=max_length)
        self.name = f"{first.name}__rerank__{Path(model).name}"

    def search(self, queries: list[str], k: int = 100) -> list[list[str]]:
        out = []
        for q, ranked in zip(queries, self.first.search(queries, max(k, self.depth)), strict=True):
            head = ranked[:self.depth]
            scores = self.encoder.predict([(q, self.articles[a].as_passage()) for a in head], batch_size=16,
                                          show_progress_bar=False)
            out.append([head[i] for i in np.argsort(-np.asarray(scores))] + ranked[self.depth:k])
        return out
