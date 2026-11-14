"""Keyword route: BM25 over the parsed text of every page.

Dense retrieval is weak on exact strings (model numbers, codes, figures); an
inverted index is not.

The analyzer follows Lucene's English analyzer, which Pyserini uses for its
standard BM25 baselines: lowercase, drop stopwords, Porter stemming. Two
changes for documents full of identifiers: compound tokens such as "TX-900" are
indexed whole and in parts, and tokens that contain digits are never stemmed.
Other ViDoRe v3 languages (French, German, Spanish, Italian, Portuguese) use the
Snowball stemmer for that language, without a stopword list.
"""

from __future__ import annotations

import functools
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import snowballstemmer

from mmrag.retrieval.base import Run

# Latin words and numbers, optionally joined by - _ . / (e.g. tx-900, v1.2, iso/iec)
_WORD = re.compile(r"[a-z0-9À-ɏ]+(?:[-_./][a-z0-9À-ɏ]+)*")
_SPLIT = re.compile(r"[-_./]")
_DIGIT = re.compile(r"\d")

# Lucene's EnglishAnalyzer stop set (the one Pyserini's BM25 baselines use).
ENGLISH_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)

# English uses the original Porter algorithm, as Lucene's PorterStemFilter does.
_STEMMERS = {
    "english": "porter",
    "french": "french",
    "german": "german",
    "spanish": "spanish",
    "italian": "italian",
    "portuguese": "portuguese",
}
LANGUAGES = tuple(_STEMMERS)


@functools.cache
def _stemmer(language: str) -> Callable[[str], str]:
    if language not in _STEMMERS:
        raise ValueError(f"Unsupported BM25 language '{language}'; choose from {LANGUAGES}")
    stemmer = snowballstemmer.stemmer(_STEMMERS[language])
    return functools.lru_cache(maxsize=200_000)(stemmer.stemWord)


def tokenize(text: str, language: str = "english") -> list[str]:
    stem = _stemmer(language)
    stopwords = ENGLISH_STOPWORDS if language == "english" else frozenset()
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        parts = [p for p in _SPLIT.split(word) if p]
        if len(parts) > 1:
            tokens.append(word)  # the whole identifier, as written
        for part in parts:
            if part in stopwords:
                continue
            tokens.append(part if _DIGIT.search(part) else stem(part))
    return tokens


class BM25Retriever:
    name = "bm25"

    def __init__(
        self,
        store_path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        language: str = "english",
    ):
        _stemmer(language)  # fail early on an unsupported language
        self.store_path = Path(store_path) if store_path else None
        self.language = language
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.avg_len = 0.0

    def index(
        self,
        corpus_ids: list[str],
        corpus_images: list[Any] | None = None,
        corpus_texts: list[str] | None = None,
        dataset_name: str | None = None,
    ) -> None:
        if corpus_texts is None:
            raise ValueError("BM25 needs page texts")
        self.ids = [str(c) for c in corpus_ids]
        lengths = []
        raw: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(corpus_texts):
            counts = Counter(tokenize(text or "", self.language))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                raw.setdefault(term, []).append((i, tf))
        self.doc_len = np.array(lengths, dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if lengths else 0.0
        self.postings = {
            term: (
                np.array([d for d, _ in plist], dtype=np.int32),
                np.array([tf for _, tf in plist], dtype=np.float32),
            )
            for term, plist in raw.items()
        }
        if self.store_path is not None:
            self.save(self.store_path)

    def _scores(self, query: str) -> np.ndarray:
        n = len(self.ids)
        scores = np.zeros(n, dtype=np.float32)
        if n == 0 or self.avg_len == 0:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avg_len)
        for term, qtf in Counter(tokenize(query, self.language)).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            docs, tf = posting
            idf = math.log(1.0 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += qtf * idf * tf * (self.k1 + 1.0) / (tf + norm[docs])
        return scores

    def search(self, query_ids: list[str], queries: list[str], top_k: int = 10) -> Run:
        run: Run = {}
        for qid, query in zip(query_ids, queries):
            scores = self._scores(query)
            order = np.argsort(-scores, kind="stable")[:top_k]
            run[str(qid)] = {self.ids[i]: float(scores[i]) for i in order if scores[i] > 0}
        return run

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "language": self.language,
            "ids": self.ids,
            "doc_len": self.doc_len.tolist(),
            "postings": {t: [d.tolist(), tf.tolist()] for t, (d, tf) in self.postings.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> BM25Retriever:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self = cls(
            store_path=path,
            k1=payload["k1"],
            b=payload["b"],
            language=payload.get("language", "english"),
        )
        self.ids = payload["ids"]
        self.doc_len = np.array(payload["doc_len"], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.postings = {
            t: (np.array(d, dtype=np.int32), np.array(tf, dtype=np.float32))
            for t, (d, tf) in payload["postings"].items()
        }
        return self

    def storage_bytes(self) -> int:
        if self.store_path is not None and self.store_path.exists():
            return self.store_path.stat().st_size
        return int(sum(d.nbytes + tf.nbytes for d, tf in self.postings.values()))

    def info(self) -> dict:
        return {
            "bm25": {
                "n_pages": len(self.ids),
                "n_terms": len(self.postings),
                "language": self.language,
            }
        }
