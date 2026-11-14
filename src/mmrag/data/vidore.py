"""Loader for ViDoRe v3 subsets (queries, page corpus, relevance judgments).

Reads the dataset's parquet files straight from the Hugging Face cache with
pyarrow, so an already-downloaded subset loads with no network access.

Files and columns:
  queries/  query_id, query, language, answer, query_types, ...
  corpus/   corpus_id, image{bytes,path}, markdown, doc_id, page_number_in_doc
  qrels/    query_id, corpus_id, score, content_type, bounding_boxes
"""

from __future__ import annotations

import io
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Qrels = dict[str, dict[str, int]]
BBoxes = dict[str, dict[str, list[dict[str, int]]]]

_PATTERNS = ["queries/*.parquet", "corpus/*.parquet", "qrels/*.parquet"]


@dataclass
class VidoreSubset:
    dataset_name: str
    query_ids: list[str]
    queries: list[str]
    corpus_ids: list[str]
    corpus_texts: list[str]
    qrels: Qrels
    # Reference answers, {query_id: answer}; empty string when the dataset has none.
    answers: dict[str, str] = field(default_factory=dict)
    # Source document and 0-based row-aligned page number of each corpus page.
    doc_ids: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    # Annotated evidence regions, {query_id: {corpus_id: [{x1,y1,x2,y2,annotator}]}}.
    bboxes: BBoxes = field(default_factory=dict)
    # Kind of evidence per judged page, {query_id: {corpus_id: ["Text", "Chart", ...]}}.
    content_types: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    snapshot_dir: Path | None = None

    def iter_page_images(
        self, corpus_ids: list[str] | None = None, batch_size: int = 16
    ) -> Iterator[tuple[str, Any]]:
        """Yield (corpus_id, PIL.Image) without holding the whole corpus in memory."""
        from PIL import Image

        for cid, raw in self.iter_page_image_bytes(corpus_ids, batch_size):
            yield cid, Image.open(io.BytesIO(raw)).convert("RGB")

    def iter_page_image_bytes(
        self, corpus_ids: list[str] | None = None, batch_size: int = 16
    ) -> Iterator[tuple[str, bytes]]:
        import pyarrow.parquet as pq

        if self.snapshot_dir is None:
            raise RuntimeError("This subset was not loaded from disk; no page images available")
        wanted = set(corpus_ids) if corpus_ids is not None else None
        if wanted is not None and not wanted:
            return
        for path in sorted((self.snapshot_dir / "corpus").glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["corpus_id", "image"]):
                for cid, image in zip(batch.column("corpus_id"), batch.column("image")):
                    cid = str(cid.as_py())
                    if wanted is None or cid in wanted:
                        yield cid, image["bytes"].as_py()

    def load_page_images(self, corpus_ids: list[str]) -> dict[str, Any]:
        return dict(self.iter_page_images(corpus_ids))


def build_qrels(rows: list[dict[str, Any]], keep_query_ids: set[str] | None = None) -> Qrels:
    """Turn qrels rows into {query_id: {corpus_id: score}}, optionally filtered by query."""
    qrels: Qrels = {}
    for row in rows:
        qid = str(row["query_id"])
        if keep_query_ids is not None and qid not in keep_query_ids:
            continue
        qrels.setdefault(qid, {})[str(row["corpus_id"])] = int(row["score"])
    return qrels


def sample_query_ids(
    query_ids: list[str],
    languages: list[str] | None,
    judged_query_ids: set[str],
    n: int | None,
    seed: int,
    language: str | None,
) -> list[str]:
    """Pick queries deterministically.

    Keeps queries in ``language`` (if given) that have at least one positive
    judgment, then draws ``n`` of them with a seeded RNG. The candidate pool is
    sorted first, so the result does not depend on dataset row order.
    """
    pool = []
    for i, qid in enumerate(query_ids):
        if language is not None and languages is not None and languages[i] != language:
            continue
        if qid not in judged_query_ids:
            continue
        pool.append(qid)
    pool = sorted(set(pool))
    if n is None or n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def resolve_snapshot(dataset_name: str) -> Path:
    """Local directory of the dataset snapshot; downloads only if it is not cached."""
    path = Path(dataset_name)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                dataset_name, repo_type="dataset", allow_patterns=_PATTERNS, local_files_only=True
            )
        )
    except Exception:
        return Path(snapshot_download(dataset_name, repo_type="dataset", allow_patterns=_PATTERNS))


def _read(snapshot: Path, part: str, columns: list[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((snapshot / part).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {snapshot / part}")
    tables = []
    for f in files:
        names = pq.ParquetFile(f).schema_arrow.names
        cols = [c for c in columns if c in names] if columns else None
        tables.append(pq.read_table(f, columns=cols))
    return pa.concat_tables(tables).to_pylist()


def load_vidore_subset(
    dataset_name: str = "vidore/vidore_v3_hr",
    n_queries: int | None = 50,
    language: str | None = "english",
    seed: int = 0,
) -> VidoreSubset:
    """Load one ViDoRe v3 subset: a seeded sample of queries and the full page corpus.

    ``dataset_name`` is a Hub dataset id or a local directory with the same layout.
    Page images are not loaded here; use ``VidoreSubset.iter_page_images``.
    """
    snapshot = resolve_snapshot(dataset_name)
    query_rows = _read(snapshot, "queries", ["query_id", "query", "language", "answer"])
    qrel_rows = _read(snapshot, "qrels", ["query_id", "corpus_id", "score", "bounding_boxes", "content_type"])
    corpus_rows = _read(
        snapshot, "corpus", ["corpus_id", "markdown", "doc_id", "page_number_in_doc"]
    )

    all_qids = [str(r["query_id"]) for r in query_rows]
    has_language = bool(query_rows) and "language" in query_rows[0]
    languages = [r["language"] for r in query_rows] if has_language else None
    full_qrels = build_qrels(qrel_rows)
    judged = {qid for qid, docs in full_qrels.items() if any(s > 0 for s in docs.values())}
    picked = sample_query_ids(all_qids, languages, judged, n_queries, seed, language)
    picked_set = set(picked)
    by_qid = {str(r["query_id"]): r for r in query_rows}

    bboxes: BBoxes = {}
    content_types: dict[str, dict[str, list[str]]] = {}
    for r in qrel_rows:
        qid = str(r["query_id"])
        if qid not in picked_set:
            continue
        if r.get("bounding_boxes"):
            bboxes.setdefault(qid, {})[str(r["corpus_id"])] = list(r["bounding_boxes"])
        if r.get("content_type"):
            content_types.setdefault(qid, {})[str(r["corpus_id"])] = list(r["content_type"])

    return VidoreSubset(
        dataset_name=dataset_name,
        query_ids=picked,
        queries=[by_qid[q]["query"] for q in picked],
        corpus_ids=[str(r["corpus_id"]) for r in corpus_rows],
        corpus_texts=[r.get("markdown") or "" for r in corpus_rows],
        qrels={q: full_qrels[q] for q in picked},
        answers={q: by_qid[q].get("answer") or "" for q in picked},
        doc_ids=[str(r.get("doc_id") or "") for r in corpus_rows],
        page_numbers=[int(r.get("page_number_in_doc") or 0) for r in corpus_rows],
        bboxes=bboxes,
        content_types=content_types,
        snapshot_dir=snapshot,
    )
