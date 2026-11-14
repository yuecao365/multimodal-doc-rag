"""Put several finished runs side by side (controls and ablations).

    uv run python -m mmrag.experiments.compare results/2026-11-14_text results/2026-11-14_hybrid

Prints a Markdown table built only from the numbers stored in each results
directory; ``--out`` also writes it to a file. Missing values are shown as "-".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mmrag.experiments.common import read_json

COLUMNS = [
    "run", "pipeline", "routes", "rerank", "visual index", "NDCG@10", "Recall@10",
    "index KB/page", "search s/query", "answer score", "citations ok",
]  # fmt: skip


def _load(path: Path, name: str):
    file = path / name
    return read_json(file) if file.exists() else None


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def summarize(results: Path) -> dict[str, str]:
    config = _load(results, "config.json") or {}
    metrics = _load(results, "metrics.json") or {}
    cost = _load(results, "cost.json") or {}
    answers = _load(results, "answer_metrics.json") or {}
    spec = config.get("pipeline", {})
    kind = spec.get("pipeline", "-")
    uses_visual = kind == "visual" or (kind == "hybrid" and "visual" in spec.get("routes", ""))
    routes = "-"
    if kind == "hybrid":
        routes = spec.get("routes", "-") + (" (all pages)" if spec.get("routing") == "all" else "")
    per_page = cost.get("index_storage_bytes_per_page")
    return {
        "run": config.get("name", results.name),
        "pipeline": kind,
        "routes": routes,
        "rerank": spec.get("rerank", "none") if kind == "hybrid" else "-",
        "visual index": (
            f"{spec.get('quantization')}, pool {spec.get('pool_factor')}" if uses_visual else "-"
        ),
        "NDCG@10": _fmt(metrics.get("ndcg_cut_10")),
        "Recall@10": _fmt(metrics.get("recall_10")),
        "index KB/page": _fmt(per_page / 1024 if per_page is not None else None, 1),
        "search s/query": _fmt(cost.get("search_seconds_per_query")),
        "answer score": _fmt((answers.get("answer") or {}).get("mean_score")),
        "citations ok": _fmt((answers.get("citations") or {}).get("answers_all_citations_ok")),
    }


def comparison_table(result_dirs: list[Path]) -> str:
    rows = [summarize(d) for d in result_dirs]
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row[c] for c in COLUMNS) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="+", help="results directories")
    p.add_argument("--out", default=None, help="also write the table to this file")
    args = p.parse_args(argv)
    table = comparison_table([Path(r) for r in args.results])
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
