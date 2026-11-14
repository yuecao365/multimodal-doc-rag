"""Stage 1 text baseline on a ViDoRe v3 subset.

    uv run python -m mmrag.experiments.baseline

Indexes the subset's page Markdown with RAG-Anything (no knowledge graph),
retrieves with LightRAG naive mode, and writes config, rankings, metrics and
cost to results/<date>_<name>/.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import logging
import platform
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from mmrag.data.vidore import load_vidore_subset
from mmrag.eval.retrieval import evaluate_run
from mmrag.models import NoKGLLM, OpenAICompatEmbedder
from mmrag.pipelines.text_baseline import TextBaselinePipeline

ROOT = Path(__file__).resolve().parents[3]


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _environment() -> dict:
    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    try:
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
    except OSError:
        commit = None
    lock = ROOT / "uv.lock"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "packages": {
            p: version(p)
            for p in ("raganything", "lightrag-hku", "datasets", "pytrec-eval-terrier", "openai")
        },
    }


async def main(args: argparse.Namespace) -> Path:
    load_dotenv(ROOT / ".env")
    date = args.date or dt.date.today().isoformat()
    out_dir = ROOT / "results" / f"{date}_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.dataset.split("/")[-1]
    embedder = OpenAICompatEmbedder.from_env()
    model_slug = embedder.model.replace("/", "_")
    working_dir = ROOT / "rag_storage" / f"{args.name}_{slug}_{model_slug}"

    config = {
        "name": args.name,
        "dataset": args.dataset,
        "split": "test",
        "language": args.language,
        "n_queries": args.n_queries,
        "seed": args.seed,
        "corpus": "full subset, page markdown",
        "indexing": "RAGAnything.insert_content_list, one document per page, no knowledge graph "
        "(LLM stub returns empty entity extraction)",
        "retrieval": "LightRAG aquery_data mode=naive, rerank off; chunks collapsed to pages "
        "in rank order",
        "embedding": {"base_url": embedder.base_url, "model": embedder.model, "dim": embedder.dim},
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap_token_size": args.chunk_overlap,
        "chunk_top_k": args.chunk_top_k,
        "top_k_pages": 10,
        "working_dir": str(working_dir.relative_to(ROOT)),
    }
    _write_json(out_dir / "config.json", config)
    _write_json(out_dir / "environment.json", _environment())

    data = load_vidore_subset(args.dataset, args.n_queries, args.language, args.seed)
    _write_json(
        out_dir / "queries.json",
        [{"query_id": q, "query": t} for q, t in zip(data.query_ids, data.queries)],
    )

    llm = NoKGLLM()
    pipeline = TextBaselinePipeline(
        working_dir=str(working_dir),
        embedder=embedder,
        embedding_dim=embedder.dim,
        embedding_max_tokens=embedder.max_tokens,
        chunk_top_k=args.chunk_top_k,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap,
        llm=llm,
    )

    t0 = time.perf_counter()
    await pipeline.aindex(data.corpus_ids, None, data.corpus_texts, args.dataset)
    index_seconds = time.perf_counter() - t0
    index_usage = embedder.usage()
    index_llm_calls = llm.calls

    t0 = time.perf_counter()
    run = await pipeline.asearch(data.query_ids, data.queries)
    search_seconds = time.perf_counter() - t0
    await pipeline.aclose()

    total_usage = embedder.usage()
    metrics, per_query = evaluate_run(data.qrels, run)

    non_empty = [t for t in data.corpus_texts if t.strip()]

    _write_json(out_dir / "run.json", run)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for qid, text in zip(data.query_ids, data.queries):
            f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "query": text,
                        **per_query[qid],
                        "n_relevant": sum(1 for s in data.qrels[qid].values() if s > 0),
                        "retrieved": list(run.get(qid, {})),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    _write_json(
        out_dir / "metrics.json",
        {
            "n_queries": len(data.query_ids),
            "n_corpus_pages": len(data.corpus_ids),
            "n_pages_skipped_empty": len(pipeline.skipped_empty),
            # Pages whose text equals an earlier page's: LightRAG keys chunks by content hash,
            # so only one page of each identical group stays retrievable.
            "n_pages_duplicate_text": len(non_empty) - len(set(non_empty)),
            "n_queries_with_fewer_than_10_pages": sum(
                1 for q in data.query_ids if len(run.get(q, {})) < 10
            ),
            **metrics,
        },
    )
    _write_json(
        out_dir / "cost.json",
        {
            "note": "Indexing figures cover only pages embedded in this invocation; a rerun over an "
            "existing working_dir skips pages already indexed.",
            "index_seconds": round(index_seconds, 1),
            "index_embedding": index_usage,
            "index_llm_calls_avoided": index_llm_calls,
            "index_storage_bytes": _dir_size(working_dir),
            "index_storage_bytes_per_page": round(
                _dir_size(working_dir) / max(1, len(data.corpus_ids))
            ),
            "search_seconds": round(search_seconds, 1),
            "search_seconds_per_query": round(search_seconds / max(1, len(data.query_ids)), 3),
            "search_embedding": {k: total_usage[k] - index_usage[k] for k in total_usage},
        },
    )
    print(json.dumps(metrics, indent=2))
    print(f"results written to {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="vidore/vidore_v3_hr")
    p.add_argument("--language", default="english")
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk-top-k", type=int, default=60)
    p.add_argument("--chunk-token-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument("--name", default="baseline")
    p.add_argument("--date", default=None, help="YYYY-MM-DD for the results dir (default: today)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(main(parse_args()))
