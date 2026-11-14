"""Optional second-stage rerankers over the fused candidates.

A reranker implements

    rerank(query, candidates, top_n) -> {page_id: score}

where ``candidates`` are one query's fused scores. It rescores the best
``top_n`` pages and returns scores for *all* candidates: reranked pages first,
in their new order, then the rest in their original order. Pages a reranker
cannot judge stay where the first stage put them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from typing import Any

from mmrag.models import image_part, text_part
from mmrag.retrieval.fusion import ranked_ids


def merge_order(new_order: list[str], candidates: dict[str, float]) -> dict[str, float]:
    """Scores that put ``new_order`` first and keep every other candidate after it."""
    seen = set(new_order)
    order = [p for p in new_order if p in candidates]
    order += [p for p in ranked_ids(candidates) if p not in seen]
    return {pid: 1.0 / (rank + 1) for rank, pid in enumerate(order)}


class MaxSimReranker:
    """Rescore candidates with full-precision late interaction.

    The first stage may search a pooled / quantized index, or may not have a page
    in the visual index at all (text pages in the routed hybrid). This reranker
    uses the uncompressed page vectors of every page, so it needs the whole corpus
    encoded; its storage is reported separately from the first-stage index.
    """

    name = "maxsim"

    def __init__(self, query_encoder: Any, page_store: Any) -> None:
        self.query_encoder = query_encoder
        self.page_store = page_store
        self.pages_scored = 0
        self.queries = 0

    def rerank(self, query: str, candidates: dict[str, float], top_n: int = 20) -> dict[str, float]:
        head = ranked_ids(candidates)[:top_n]
        vectors = self.query_encoder.encode_queries([query])[0]
        scores = self.page_store.score_pages(vectors, head)
        self.queries += 1
        self.pages_scored += len(scores)
        judged = sorted(scores, key=lambda p: (-scores[p], p))
        # Candidates without vectors keep their first-stage position after the judged ones.
        return merge_order(judged, candidates)

    def storage_bytes(self) -> int:
        return int(self.page_store.nbytes)

    def info(self) -> dict:
        return {
            "reranker": self.name,
            "queries": self.queries,
            "pages_scored": self.pages_scored,
            "storage_bytes": self.storage_bytes(),
        }


LLM_RERANK_SYSTEM = """You rank document pages by how useful they are for answering a question.

Each page is introduced by a line "[PAGE <id>]" followed by an excerpt of its text or its image.
Rank every page, most useful first. Use only the ids given.
Reply with one JSON object and nothing else: {"ranking": ["<id>", "<id>", ...]}"""


class LLMReranker:
    """Listwise reranking by a chat / vision model: one call per query.

    Pages are shown as a text excerpt; a page with no text is shown as its image
    when images are available. ``prepare`` is called by the pipeline at indexing
    time with the corpus.
    """

    name = "llm"

    def __init__(self, client: Any, excerpt_chars: int = 1200) -> None:
        self.client = client
        self.excerpt_chars = excerpt_chars
        self.texts: dict[str, str] = {}
        self.image_provider: Callable[[list[str]], Iterable[tuple[str, Any]]] | None = None
        self.queries = 0
        self.unparsed = 0

    def prepare(self, corpus_ids, corpus_texts, image_provider=None) -> None:
        self.texts = dict(zip(map(str, corpus_ids), corpus_texts))
        self.image_provider = image_provider

    def _parts(self, query: str, page_ids: list[str]) -> list[dict]:
        need_image = [p for p in page_ids if not (self.texts.get(p) or "").strip()]
        images = {}
        if need_image and self.image_provider is not None:
            images = dict(self.image_provider(need_image))
        parts: list[dict] = []
        for pid in page_ids:
            parts.append(text_part(f"[PAGE {pid}]"))
            text = (self.texts.get(pid) or "").strip()
            if text:
                parts.append(text_part(text[: self.excerpt_chars]))
            elif pid in images:
                parts.append(image_part(images[pid], max_side=1024))
            else:
                parts.append(text_part("(no text available)"))
        parts.append(text_part(f"Question: {query}"))
        return parts

    def rerank(self, query: str, candidates: dict[str, float], top_n: int = 20) -> dict[str, float]:
        head = ranked_ids(candidates)[:top_n]
        if len(head) < 2:
            return merge_order(head, candidates)
        raw = self.client.complete(LLM_RERANK_SYSTEM, self._parts(query, head), json_mode=True)
        self.queries += 1
        ranking = self._parse(raw, set(head))
        if ranking is None:
            self.unparsed += 1
            return merge_order(head, candidates)  # keep the first-stage order
        # Pages the model left out follow, in first-stage order.
        ranking += [p for p in head if p not in set(ranking)]
        return merge_order(ranking, candidates)

    @staticmethod
    def _parse(raw: str, allowed: set[str]) -> list[str] | None:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            ranking = json.loads(match.group(0)).get("ranking")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(ranking, list):
            return None
        out: list[str] = []
        for item in ranking:
            pid = re.sub(r"^\[?PAGE\s*|\]$", "", str(item).strip(), flags=re.IGNORECASE).strip()
            if pid in allowed and pid not in out:
                out.append(pid)
        return out or None

    def storage_bytes(self) -> int:
        return 0

    def info(self) -> dict:
        info = {"reranker": self.name, "queries": self.queries, "unparsed_replies": self.unparsed}
        if hasattr(self.client, "usage"):
            info["usage"] = self.client.usage()
        return info
