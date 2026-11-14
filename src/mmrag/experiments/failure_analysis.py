"""Failure analysis of a finished run: where did the relevant pages get lost?

    uv run python -m mmrag.experiments.failure_analysis --results results/2026-11-14_hybrid

Uses only files already in the results directory (plus the dataset for the
relevance judgments); no model is called. Writes failure_analysis.json and a
readable failure_analysis.md next to them.

For every relevant page that is not in the final top-k it records the stage
that lost it:

  not_retrieved_by_any_route   no route had it among its candidates
  ranked_too_low_after_fusion  some route found it, fusion left it below top-k
  demoted_by_rerank            it was in the fused top-k, the reranker pushed it out
  not_in_top_k                 single-route run: simply ranked too low

and, when answers were generated, whether a wrong answer is a retrieval
failure (no relevant page was shown) or a generation failure (one was).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from mmrag.data.vidore import load_vidore_subset
from mmrag.experiments.common import read_json, read_jsonl, write_json
from mmrag.retrieval.fusion import ranked_ids

ROUTE_NAMES = ("text", "bm25", "visual")


def _rank(run: dict, qid: str, page: str) -> int | None:
    order = ranked_ids(run.get(qid, {}))
    return order.index(page) + 1 if page in order else None


def analyze_failures(results: Path) -> dict:
    config = read_json(results / "config.json")
    data = load_vidore_subset(
        config["dataset"], config["n_queries"], config["language"], config["seed"]
    )
    top_k = config["pipeline"]["top_k"]
    run = read_json(results / "run.json")
    route_runs = {
        name: read_json(results / f"run_{name}.json")
        for name in ROUTE_NAMES
        if (results / f"run_{name}.json").exists()
    }
    fused = read_json(results / "run_fused.json") if (results / "run_fused.json").exists() else None
    page_routes = (
        read_json(results / "page_routes.json") if (results / "page_routes.json").exists() else {}
    )
    reranked = config["pipeline"].get("rerank", "none") != "none"

    misses, per_query = [], []
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # content type -> [hit, total]
    by_route: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # page route -> [hit, total]
    only_found_by: Counter = Counter()

    for qid, query in zip(data.query_ids, data.queries):
        relevant = sorted(p for p, s in data.qrels[qid].items() if s > 0)
        final = ranked_ids(run.get(qid, {}))[:top_k]
        missed = [p for p in relevant if p not in final]
        for page in relevant:
            hit = page in final
            for ctype in data.content_types.get(qid, {}).get(page, ["unknown"]):
                by_type[ctype][0] += hit
                by_type[ctype][1] += 1
            route = page_routes.get(page, "n/a")
            by_route[route][0] += hit
            by_route[route][1] += 1
            found_by = [n for n, r in route_runs.items() if (_rank(r, qid, page) or 1e9) <= top_k]
            if len(found_by) == 1:
                only_found_by[found_by[0]] += 1

        for page in missed:
            ranks = {name: _rank(r, qid, page) for name, r in route_runs.items()}
            fused_rank = _rank(fused, qid, page) if fused is not None else None
            if not route_runs:
                stage = "not_in_top_k"
            elif all(rank is None for rank in ranks.values()):
                stage = "not_retrieved_by_any_route"
            elif reranked and fused_rank is not None and fused_rank <= top_k:
                stage = "demoted_by_rerank"
            else:
                stage = "ranked_too_low_after_fusion"
            misses.append(
                {
                    "query_id": qid,
                    "page": page,
                    "stage": stage,
                    "page_route": page_routes.get(page),
                    "content_types": data.content_types.get(qid, {}).get(page, []),
                    "route_ranks": ranks,
                    "fused_rank": fused_rank,
                    "final_rank": _rank(run, qid, page),
                }
            )
        per_query.append(
            {
                "query_id": qid,
                "query": query,
                "n_relevant": len(relevant),
                "n_found": len(relevant) - len(missed),
                "missed": missed,
            }
        )

    summary = {
        "name": config["name"],
        "top_k": top_k,
        "n_queries": len(per_query),
        "n_relevant_pages": sum(q["n_relevant"] for q in per_query),
        "n_missed_pages": len(misses),
        "queries_with_no_relevant_page_found": sum(1 for q in per_query if q["n_found"] == 0),
        "queries_with_all_relevant_pages_found": sum(1 for q in per_query if not q["missed"]),
        "misses_by_stage": dict(Counter(m["stage"] for m in misses)),
        "page_recall_by_evidence_type": {
            t: {"found": h, "total": n, "recall": round(h / n, 3)}
            for t, (h, n) in sorted(by_type.items())
        },
        "page_recall_by_page_route": {
            r: {"found": h, "total": n, "recall": round(h / n, 3)}
            for r, (h, n) in sorted(by_route.items())
        },
        # Relevant pages that exactly one route had in its own top-k: what each route adds.
        "relevant_pages_found_by_one_route_only": dict(only_found_by),
    }

    answers_file = results / "answers.jsonl"
    if answers_file.exists():
        summary["answers"] = _answer_failures(read_jsonl(answers_file))

    worst = sorted(per_query, key=lambda q: (q["n_found"] / max(1, q["n_relevant"]), q["query_id"]))
    return {"summary": summary, "misses": misses, "queries": worst}


def _answer_failures(records: list[dict]) -> dict:
    causes: Counter = Counter()
    examples = []
    for r in records:
        if r.get("judge_verdict") in (None, "correct"):
            continue
        if not r["relevant_pages_in_context"]:
            cause = "retrieval: no relevant page shown"
        elif not r["provenance"]["cites_a_relevant_page"]:
            cause = "generation: relevant page shown but not cited"
        else:
            cause = "generation: relevant page shown and cited"
        causes[cause] += 1
        examples.append({"query_id": r["query_id"], "verdict": r["judge_verdict"], "cause": cause})
    citation_problems = sum(
        1
        for r in records
        if r["citations"]["page_not_in_context"] or r["citations"]["quote_not_found"]
    )
    return {
        "n_answers": len(records),
        "n_not_correct": len(examples),
        "causes": dict(causes),
        "answers_with_failed_citation_checks": citation_problems,
        "not_correct": examples,
    }


def to_markdown(report: dict, max_queries: int = 15) -> str:
    s = report["summary"]
    lines = [
        f"# Failure analysis: {s['name']}",
        "",
        (
            f"- Queries: {s['n_queries']}; relevant pages: {s['n_relevant_pages']}; "
            f"missed from top-{s['top_k']}: {s['n_missed_pages']}"
        ),
        f"- Queries with no relevant page found: {s['queries_with_no_relevant_page_found']}",
        f"- Queries with every relevant page found: {s['queries_with_all_relevant_pages_found']}",
        "",
        "## Where relevant pages were lost",
        "",
        "| Stage | Pages |",
        "|---|---|",
        *[f"| {stage} | {n} |" for stage, n in sorted(s["misses_by_stage"].items())],
        "",
        "## Page recall by evidence type",
        "",
        "| Evidence | Found | Total | Recall |",
        "|---|---|---|---|",
        *[
            f"| {t} | {v['found']} | {v['total']} | {v['recall']} |"
            for t, v in s["page_recall_by_evidence_type"].items()
        ],
        "",
        "## Page recall by route the page was sent to",
        "",
        "| Route | Found | Total | Recall |",
        "|---|---|---|---|",
        *[
            f"| {r} | {v['found']} | {v['total']} | {v['recall']} |"
            for r, v in s["page_recall_by_page_route"].items()
        ],
    ]
    if s["relevant_pages_found_by_one_route_only"]:
        lines += [
            "",
            "## Relevant pages only one route had in its top-k",
            "",
            "| Route | Pages |",
            "|---|---|",
            *[
                f"| {r} | {n} |"
                for r, n in sorted(s["relevant_pages_found_by_one_route_only"].items())
            ],
        ]
    if "answers" in s:
        a = s["answers"]
        lines += [
            "",
            "## Answers",
            "",
            f"- Not graded correct: {a['n_not_correct']} of {a['n_answers']}",
            f"- Answers with a failed citation check: {a['answers_with_failed_citation_checks']}",
            "",
            "| Cause | Answers |",
            "|---|---|",
            *[f"| {c} | {n} |" for c, n in sorted(a["causes"].items())],
        ]
    lines += ["", "## Queries with the lowest page recall", ""]
    for q in report["queries"][:max_queries]:
        if not q["missed"]:
            break
        lines.append(
            f"- `{q['query_id']}` found {q['n_found']}/{q['n_relevant']}, "
            f"missed pages {', '.join(q['missed'])}: {q['query']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True)
    args = p.parse_args(argv)
    results = Path(args.results)
    report = analyze_failures(results)
    write_json(results / "failure_analysis.json", report)
    (results / "failure_analysis.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
