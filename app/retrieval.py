"""Recuperación de información: BM25 léxico + fusión opcional con embeddings."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .models import Chunk, SourceRef

WORD = re.compile(r"[a-z0-9áéíóúüñ]+")

STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "u",
    "en", "por", "para", "con", "sin", "que", "del", "al", "se", "es", "son",
    "como", "mas", "más", "pero", "sus", "su", "lo", "le", "les", "ya", "hay",
    "the", "a", "an", "of", "to", "in", "and", "is", "are", "for", "on", "with",
}


def tokenize(text: str) -> list[str]:
    """Minúsculas, sin acentos y sin palabras vacías."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return [w for w in WORD.findall(text) if w not in STOPWORDS and len(w) > 1]


@dataclass
class Scored:
    chunk: Chunk
    score: float


class BM25Index:
    """Implementación compacta de BM25 (k1=1.5, b=0.75) sin dependencias externas."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(c.text) for c in chunks]
        self.freqs = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[Scored]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []

        scores: list[float] = []
        for freq, length in zip(self.freqs, self.lengths):
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                norm = tf + self.k1 * (1 - self.b + self.b * (length / (self.avg_len or 1)))
                score += idf * (tf * (self.k1 + 1)) / norm
            scores.append(score)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            Scored(self.chunks[i], round(scores[i], 4))
            for i in ranked[:top_k]
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(
    rankings: list[list[str]], weights: list[float] | None = None, k: int = 60
) -> dict[str, float]:
    """RRF: combina varias listas ordenadas de ids en una sola puntuación."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for position, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + position)
    return dict(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))


def to_sources(scored: list[Scored], limit: int = 320) -> list[SourceRef]:
    return [
        SourceRef(
            chunk_id=s.chunk.id,
            filename=s.chunk.filename,
            page=s.chunk.page,
            score=s.score,
            excerpt=(s.chunk.text[:limit] + "…") if len(s.chunk.text) > limit else s.chunk.text,
        )
        for s in scored
    ]
