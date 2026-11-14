"""Keyword route: BM25 over the parsed text of every page.

Dense retrieval is weak on exact strings (model numbers, codes, figures); an
inverted index is not. Chinese text is segmented with jieba, everything else is
split on non-alphanumerics, and compound tokens such as "TX-900" are indexed
both whole and in parts.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mmrag.retrieval.base import Run

_CJK_RUN = re.compile(r"[㐀-䶿一-鿿]+")
_KANA_HANGUL = re.compile(r"[぀-ヿ가-힯]")
# Latin/digit words, optionally joined by - _ . / (e.g. tx-900, v1.2, iso/iec)
_WORD = re.compile(r"[a-z0-9À-ɏ]+(?:[-_./][a-z0-9À-ɏ]+)*")
_SPLIT = re.compile(r"[-_./]")

_jieba = None


def _cut_chinese(text: str) -> list[str]:
    global _jieba
    if _jieba is None:
        try:
            import jieba

            jieba.setLogLevel(60)
            _jieba = jieba
        except ImportError:  # pragma: no cover - jieba is a declared dependency
            _jieba = False
    if _jieba:
        return [t for t in _jieba.lcut_for_search(text) if t.strip()]
    # Fallback: overlapping character bigrams.
    return [text[i : i + 2] for i in range(max(1, len(text) - 1))]


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for word in _WORD.findall(text):
        tokens.append(word)
        parts = _SPLIT.split(word)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    for run in _CJK_RUN.findall(text):
        tokens.extend(_cut_chinese(run))
    tokens.extend(_KANA_HANGUL.findall(text))
    return tokens


class BM25Retriever:
    name = "bm25"

    def __init__(self, store_path: str | Path | None = None, k1: float = 1.5, b: float = 0.75):
        self.store_path = Path(store_path) if store_path else None
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
            counts = Counter(tokenize(text or ""))
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
        for term, qtf in Counter(tokenize(query)).items():
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
            "ids": self.ids,
            "doc_len": self.doc_len.tolist(),
            "postings": {t: [d.tolist(), tf.tolist()] for t, (d, tf) in self.postings.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> BM25Retriever:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self = cls(store_path=path, k1=payload["k1"], b=payload["b"])
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
        return {"bm25": {"n_pages": len(self.ids), "n_terms": len(self.postings)}}
