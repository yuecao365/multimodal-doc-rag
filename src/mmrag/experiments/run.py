"""Run one retrieval experiment on a ViDoRe v3 subset.

    uv run python -m mmrag.experiments.run --pipeline text

Indexes the subset, retrieves for a seeded sample of queries, and writes the
configuration, rankings, metrics and cost to results/<date>_<name>/.
See ``--help`` for the pipeline choices and ablation switches.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from mmrag.data.vidore import load_vidore_subset
from mmrag.eval.retrieval import evaluate_run
from mmrag.experiments.common import ROOT, environment, results_dir, write_json, write_jsonl
from mmrag.pipelines.factory import PIPELINES, PipelineSpec, build_pipeline

logger = logging.getLogger("mmrag.run")


def add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose every PipelineSpec field as a command-line option."""
    for f in dataclasses.fields(PipelineSpec):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "pipeline":
            parser.add_argument(flag, default=f.default, choices=PIPELINES)
        elif f.type in ("bool", bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.type in ("int", int):
            parser.add_argument(flag, type=int, default=f.default)
        elif f.type in ("float", float):
            parser.add_argument(flag, type=float, default=f.default)
        else:
            parser.add_argument(flag, default=f.default)


def spec_from_args(args: argparse.Namespace) -> PipelineSpec:
    return PipelineSpec(**{f.name: getattr(args, f.name) for f in dataclasses.fields(PipelineSpec)})


def run_experiment(args: argparse.Namespace, embedder=None) -> Path:
    load_dotenv(ROOT / ".env")
    spec = spec_from_args(args)
    name = args.name or spec.pipeline
    out_dir = results_dir(name, args.date)
    slug = args.dataset.replace("\\", "/").rstrip("/").split("/")[-1]
    work_dir = Path(args.work_dir) if args.work_dir else ROOT / "rag_storage" / f"{name}_{slug}"

    data = load_vidore_subset(args.dataset, args.n_queries, args.language, args.seed)
    config = {
        "name": name,
        "dataset": args.dataset,
        "language": args.language,
        "n_queries": args.n_queries,
        "seed": args.seed,
        "corpus": "full subset",
        "work_dir": str(work_dir),
        "pipeline": spec.to_dict(),
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "environment.json", environment())
    write_json(
        out_dir / "queries.json",
        [{"query_id": q, "query": t} for q, t in zip(data.query_ids, data.queries)],
    )

    pipeline = build_pipeline(spec, work_dir, embedder=embedder, data=data)

    t0 = time.perf_counter()
    pipeline.index(data.corpus_ids, None, data.corpus_texts, args.dataset)
    index_seconds = time.perf_counter() - t0
    index_info = _info(pipeline)

    t0 = time.perf_counter()
    run = pipeline.search(data.query_ids, data.queries, top_k=spec.top_k)
    search_seconds = time.perf_counter() - t0
    info = _info(pipeline)
    storage = pipeline.storage_bytes()
    if hasattr(pipeline, "close"):
        pipeline.close()

    metrics, per_query = evaluate_run(data.qrels, run)
    non_empty = [t for t in data.corpus_texts if t.strip()]
    write_json(out_dir / "run.json", run)
    for route, route_run in (info.pop("route_runs", None) or {}).items():
        write_json(out_dir / f"run_{route}.json", route_run)
    if "page_routes" in info:
        write_json(out_dir / "page_routes.json", info.pop("page_routes"))
    write_jsonl(
        out_dir / "per_query.jsonl",
        [
            {
                "query_id": qid,
                "query": text,
                **per_query[qid],
                "n_relevant": sum(1 for s in data.qrels[qid].values() if s > 0),
                "retrieved": list(run.get(qid, {})),
            }
            for qid, text in zip(data.query_ids, data.queries)
        ],
    )
    write_json(
        out_dir / "metrics.json",
        {
            "n_queries": len(data.query_ids),
            "n_corpus_pages": len(data.corpus_ids),
            "n_pages_empty_text": len(data.corpus_texts) - len(non_empty),
            # Pages whose text equals an earlier page's: LightRAG keys chunks by content
            # hash, so the text route keeps only one page of each identical group.
            "n_pages_duplicate_text": len(non_empty) - len(set(non_empty)),
            "n_queries_with_fewer_than_top_k": sum(
                1 for q in data.query_ids if len(run.get(q, {})) < spec.top_k
            ),
            **metrics,
        },
    )
    n_pages, n_queries = max(1, len(data.corpus_ids)), max(1, len(data.query_ids))
    write_json(
        out_dir / "cost.json",
        {
            "note": "Indexing figures cover only work done in this invocation; a rerun over "
            "an existing work_dir reuses what is already indexed.",
            "index_seconds": round(index_seconds, 2),
            "index_storage_bytes": storage,
            "index_storage_bytes_per_page": round(storage / n_pages),
            "search_seconds": round(search_seconds, 2),
            "search_seconds_per_query": round(search_seconds / n_queries, 4),
            "after_index": index_info,
            "after_search": info,
        },
    )
    logger.info("metrics: %s", metrics)
    print(f"{name}: {metrics}\nresults written to {out_dir}")
    return out_dir


def _info(pipeline) -> dict:
    """Usage counters a pipeline chooses to report (model calls, tokens, per-route detail)."""
    info = pipeline.info() if hasattr(pipeline, "info") else {}
    embedder = getattr(pipeline, "embedder", None)
    if embedder is not None and hasattr(embedder, "usage"):
        info = {**info, "text_embedding": embedder.usage()}
    llm = getattr(pipeline, "llm", None)
    if llm is not None and hasattr(llm, "calls"):
        info = {**info, "kg_llm_calls_avoided": llm.calls}
    return info


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default="vidore/vidore_v3_hr", help="Hub id or local directory")
    p.add_argument("--language", default="english")
    p.add_argument("--n-queries", type=int, default=50, help="0 = all queries")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="results/<date>_<name>; default: pipeline name")
    p.add_argument("--date", default=None, help="YYYY-MM-DD for the results dir (default: today)")
    p.add_argument("--work-dir", default=None, help="where indexes are stored")
    add_spec_arguments(p)
    return p


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.n_queries == 0:
        args.n_queries = None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return run_experiment(args)


if __name__ == "__main__":
    main()
