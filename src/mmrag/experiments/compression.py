"""Index size against retrieval quality, for the text and the visual index.

    uv run python -m mmrag.experiments.compression --visual-store data/vectors/vidore_v3_hr
    uv run python -m mmrag.experiments.compression --render results/<date>_<name>

Text index: Matryoshka truncation of the embeddings (1536 / 768 / 512 / 256 / 128).
Visual index: quantization (float16 / int8 / binary) and token pooling (1 / 2 / 3),
each with and without the full-precision MaxSim rerank of the top 20.

Each setting is one retrieval run of that route on its own, compared with the
uncompressed route by a paired test. Nothing is re-encoded: text embeddings come
from the embedding cache, visual vectors from the stored full-precision ones.
Writes table.json, table.md and chart.svg to results/<date>_<name>/. The chosen
setting of each index is the smallest one that is not significantly worse; reranked
settings are shown but not chosen, because the rerank needs the full vectors kept.
table.json holds one row per setting (metrics, storage, test result, usage
counters) and no per-query or per-page data; ``--render`` rebuilds table.md and
chart.svg from it without running anything.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from mmrag.data.vidore import load_vidore_subset
from mmrag.eval.retrieval import evaluate_run
from mmrag.eval.significance import paired_test
from mmrag.experiments import spend
from mmrag.experiments.chart import line_chart
from mmrag.experiments.common import (
    ROOT,
    environment,
    read_json,
    results_dir,
    work_root,
    write_json,
)
from mmrag.pipelines.factory import PipelineSpec, build_pipeline

TEXT_DIMS = (0, 768, 512, 256, 128)  # 0 = full
QUANTIZATIONS = ("float16", "int8", "binary")
POOL_FACTORS = (1, 2, 3)
# Per-query and per-page detail in a pipeline's report; 23 settings of it made a 24 MB table.
DETAIL_KEYS = ("route_runs", "page_routes")


def settings(visual_store: str, text_dims=TEXT_DIMS) -> list[tuple[str, str, PipelineSpec]]:
    """(index, label, spec) of every setting; the first of each index is uncompressed."""
    out = [
        ("text", f"{dim or 'full'} dims", PipelineSpec(pipeline="text", embedding_dim=dim))
        for dim in text_dims
    ]
    for rerank in ("none", "maxsim"):
        for quantization in QUANTIZATIONS:
            for pool in POOL_FACTORS:
                spec = PipelineSpec(
                    pipeline="hybrid", routes="visual", routing="all", visual_store=visual_store,
                    quantization=quantization, pool_factor=pool, rerank=rerank,
                )  # fmt: skip
                index = "visual" if rerank == "none" else "visual + rerank"
                out.append((index, f"{quantization}, pool {pool}", spec))
    return out


def measure(spec: PipelineSpec, data, work_dir, **models) -> dict:
    pipeline = build_pipeline(spec, work_dir, data=data, **models)
    pipeline.index(data.corpus_ids, None, data.corpus_texts, data.dataset_name)
    run = pipeline.search(data.query_ids, data.queries, top_k=spec.top_k)
    storage = pipeline.storage_bytes()
    usage = pipeline.info() if hasattr(pipeline, "info") else {}
    usage = {k: v for k, v in usage.items() if k not in DETAIL_KEYS}
    if hasattr(pipeline, "embedder") and hasattr(pipeline.embedder, "usage"):
        usage = {**usage, "embedding": pipeline.embedder.usage()}
    if hasattr(pipeline, "close"):
        pipeline.close()
    metrics, per_query = evaluate_run(data.qrels, run)
    return {
        "ndcg_cut_10": metrics["ndcg_cut_10"],
        "recall_10": metrics["recall_10"],
        "kb_per_page": storage / len(data.corpus_ids) / 1024,
        "per_query": {q: s["ndcg_cut_10"] for q, s in per_query.items()},
        "usage": usage,
    }


def compression_table(
    data, visual_store: str, work_dir, text_dims=TEXT_DIMS, **models
) -> list[dict]:
    rows, reference = [], {}
    for index, label, spec in settings(visual_store, text_dims):
        result = measure(spec, data, work_dir, **models)
        base = reference.setdefault(index.split(" + ")[0], result)  # the uncompressed route
        test = paired_test(result["per_query"], base["per_query"])
        rows.append(
            {
                "index": index,
                "setting": label,
                "ndcg_cut_10": round(result["ndcg_cut_10"], 4),
                "recall_10": round(result["recall_10"], 4),
                "kb_per_page": round(result["kb_per_page"], 2),
                "difference": round(test["mean"], 4),
                "ci95": [round(v, 4) for v in test["ci95"]],
                "p_value": round(test["p_value"], 4),
                "worse": bool(test["mean"] < 0 and test["p_value"] < 0.05),
                "spec": spec.to_dict(),
                "usage": result["usage"],
            }
        )
        logging.getLogger("mmrag.compression").info("%s", {**rows[-1], "spec": "", "usage": ""})
    for index in ("text", "visual"):
        ok = [r for r in rows if r["index"] == index and not r["worse"]]
        chosen = min(ok, key=lambda r: r["kb_per_page"])
        for row in rows:
            row["chosen"] = row.get("chosen", False) or row is chosen
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| index | setting | NDCG@10 | Recall@10 | KB/page | difference [95% CI] | p | chosen |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        low, high = r["ci95"]
        lines.append(
            f"| {r['index']} | {r['setting']} | {r['ndcg_cut_10']:.3f} | {r['recall_10']:.3f} "
            f"| {r['kb_per_page']:.1f} | {r['difference']:+.3f} [{low:+.3f}, {high:+.3f}] "
            f"| {r['p_value']:.4f} | {'yes' if r['chosen'] else ''} |"
        )
    return "\n".join(lines) + "\n"


def render(out: Path) -> str:
    """table.md and chart.svg from the table.json in ``out``."""
    rows = read_json(out / "table.json")
    dataset = read_json(out / "config.json")["dataset"]
    (out / "table.md").write_text(to_markdown(rows), encoding="utf-8")
    series: dict[str, list[tuple[float, float, str]]] = {}
    for r in rows:
        series.setdefault(r["index"], []).append((r["kb_per_page"], r["ndcg_cut_10"], r["setting"]))
    line_chart(
        series, out / "chart.svg", f"Index size vs retrieval quality ({dataset})",
        "index size, KB per page (log scale)", "NDCG@10",
    )  # fmt: skip
    return to_markdown(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default="vidore/vidore_v3_hr")
    p.add_argument("--visual-store", help="directory of the page and query vectors")
    p.add_argument("--name", default="compression")
    p.add_argument("--work-dir", help="index directory under the work root")
    p.add_argument("--date", default=None)
    p.add_argument("--render", help="results directory: rebuild table.md and chart.svg only")
    args = p.parse_args(argv)
    if args.render:
        print(render(Path(args.render)))
        return
    if not (args.visual_store and args.work_dir):
        p.error("--visual-store and --work-dir are required for a run")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_dotenv(ROOT / ".env")
    spend.check_budget()

    data = load_vidore_subset(args.dataset)
    out = results_dir(args.name, args.date)
    write_json(out / "config.json", {"dataset": args.dataset, "visual_store": args.visual_store})
    write_json(out / "environment.json", environment())
    rows = compression_table(data, args.visual_store, work_root() / args.work_dir)
    spend.record(out.name, "text embeddings for the compression grid", [r["usage"] for r in rows])
    write_json(out / "table.json", rows)
    print(render(out))


if __name__ == "__main__":
    main()
