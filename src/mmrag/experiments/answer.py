"""Generate and grade answers for a finished retrieval run.

    uv run python -m mmrag.experiments.answer --results results/2026-11-14_hybrid

Reads the run's rankings, shows the top pages to the VLM (as images or text),
checks the citations in code, grades the answer against the dataset's reference
answer, and writes into the same results directory:

    answers.jsonl          one record per query (answer, citations, checks, grade)
    answer_metrics.json    answer quality, citation validity, provenance
    answer_cost.json       model calls, tokens, latency
    review_sample.csv      a sample for a person to label, to check the grader

This calls the chat model configured in .env (LLM_*). It costs money.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mmrag.data.vidore import load_vidore_subset
from mmrag.eval.answers import export_review_sample, judge_answer, summarize_judgments
from mmrag.eval.provenance import citation_page_metrics, mean_of, region_iou
from mmrag.experiments.common import ROOT, read_json, write_json, write_jsonl
from mmrag.generation.answer import generate_answer
from mmrag.generation.citations import summarize_reports, verify_citations
from mmrag.generation.context import assemble_context
from mmrag.retrieval.fusion import ranked_ids

logger = logging.getLogger("mmrag.answer")


def page_forms(results: Path, pipeline: str, form: str, page_ids: list[str]) -> dict[str, str]:
    """How each page is shown to the model.

    "auto" follows the retrieval route: pages the analyzer sent to the visual route
    are shown as images, text pages as text; a visual-only run shows images, a
    text-only run shows text.
    """
    if form in ("image", "text"):
        return dict.fromkeys(page_ids, form)
    routes_file = results / "page_routes.json"
    if routes_file.exists():
        routes = read_json(routes_file)
        return {p: "text" if routes.get(p) == "text" else "image" for p in page_ids}
    default = "image" if pipeline == "visual" else "text"
    return dict.fromkeys(page_ids, default)


def answer_run(
    results: Path,
    generator: Any,
    judge: Any | None = None,
    max_pages: int = 5,
    token_budget: int = 12_000,
    form: str = "auto",
    with_regions: bool = False,
    limit: int | None = None,
    review_n: int = 20,
) -> dict:
    config = read_json(results / "config.json")
    run = read_json(results / "run.json")
    data = load_vidore_subset(
        config["dataset"], config["n_queries"], config["language"], config["seed"]
    )
    texts = dict(zip(data.corpus_ids, data.corpus_texts))
    labels = {
        cid: f"{doc} p.{page + 1}"
        for cid, doc, page in zip(data.corpus_ids, data.doc_ids, data.page_numbers)
    }
    forms = page_forms(results, config["pipeline"]["pipeline"], form, data.corpus_ids)

    records, reports, judgments, provenance, region_scores = [], [], [], [], []
    generate_seconds = 0.0
    query_ids = data.query_ids[:limit] if limit else data.query_ids
    for i, qid in enumerate(query_ids, start=1):
        query = data.queries[data.query_ids.index(qid)]
        ranked = ranked_ids(run.get(qid, {}))
        head = ranked[:max_pages]
        need_images = [p for p in head if forms[p] == "image" or not texts[p].strip()]
        images = data.load_page_images(need_images) if need_images else {}
        contexts = assemble_context(
            ranked, texts, images, forms, max_pages=max_pages, token_budget=token_budget,
            labels=labels,
        )  # fmt: skip

        t0 = time.perf_counter()
        answer = generate_answer(generator, query, contexts, with_regions=with_regions)
        generate_seconds += time.perf_counter() - t0

        shown = [c.page_id for c in contexts]
        report = verify_citations(answer, shown, texts)
        reports.append(report)
        prov = citation_page_metrics(report.valid_pages, data.qrels[qid])
        provenance.append(prov)

        regions = []
        if with_regions:
            for citation in answer.citations:
                gold = data.bboxes.get(qid, {}).get(citation.page)
                if citation.region and gold and citation.page in images:
                    score = region_iou(citation.region, gold, images[citation.page].size)
                    regions.append({"page": citation.page, "iou": round(score, 3)})
                    region_scores.append(score)

        reference = data.answers.get(qid, "")
        judgment = None
        if judge is not None and reference:
            judgment = judge_answer(judge, query, reference, answer.answer)
            judgments.append(judgment)

        records.append(
            {
                "query_id": qid,
                "question": query,
                "reference": reference,
                "answer": answer.answer,
                "answer_parsed": answer.parsed,
                "context": [
                    {
                        "page": c.page_id,
                        "form": c.form,
                        "tokens": c.tokens,
                        "truncated": c.truncated,
                    }
                    for c in contexts
                ],
                "relevant_pages_in_context": sorted(set(shown) & _relevant(data.qrels[qid])),
                "citations": report.to_dict(),
                "provenance": prov,
                "regions": regions,
                "judge_verdict": judgment["verdict"] if judgment else None,
                "judge_score": judgment["score"] if judgment else None,
                "judge_reason": judgment["reason"] if judgment else None,
            }
        )
        if i % 10 == 0 or i == len(query_ids):
            logger.info("answered %d/%d", i, len(query_ids))

    write_jsonl(results / "answers.jsonl", records)
    n = max(1, len(records))
    metrics = {
        "n_queries": len(records),
        "settings": {
            "max_pages": max_pages,
            "token_budget": token_budget,
            "form": form,
            "with_regions": with_regions,
        },
        "answers_unparsed": sum(1 for r in records if not r["answer_parsed"]),
        "answer": summarize_judgments(judgments) if judge is not None else None,
        "citations": summarize_reports(reports),
        "provenance": mean_of(provenance),
        "context_has_relevant_page": sum(1 for r in records if r["relevant_pages_in_context"]) / n,
        "region": {
            "n_scored": len(region_scores),
            "mean_iou": sum(region_scores) / len(region_scores) if region_scores else None,
        },
    }
    write_json(results / "answer_metrics.json", metrics)
    write_json(
        results / "answer_cost.json",
        {
            "generate_seconds": round(generate_seconds, 2),
            "generate_seconds_per_query": round(generate_seconds / n, 3),
            "context_tokens_estimated_per_query": round(
                sum(c["tokens"] for r in records for c in r["context"]) / n
            ),
            "images_per_query": sum(
                1 for r in records for c in r["context"] if c["form"] == "image"
            )
            / n,
            "generator": generator.usage() if hasattr(generator, "usage") else None,
            "judge": judge.usage() if judge is not None and hasattr(judge, "usage") else None,
        },
    )
    if judge is not None:
        export_review_sample(records, results / "review_sample.csv", n=review_n)
    return metrics


def _relevant(qrels: dict[str, int]) -> set[str]:
    return {p for p, score in qrels.items() if score > 0}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", required=True, help="results directory of a retrieval run")
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--token-budget", type=int, default=12_000)
    p.add_argument("--form", default="auto", choices=("auto", "image", "text"))
    p.add_argument("--regions", action="store_true", help="also ask for evidence regions")
    p.add_argument("--no-judge", action="store_true", help="generate answers without grading")
    p.add_argument("--limit", type=int, default=None, help="only the first N queries")
    p.add_argument("--review-n", type=int, default=20, help="size of the human review sample")
    args = p.parse_args(argv)

    from mmrag.models import ChatClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_dotenv(ROOT / ".env")
    generator = ChatClient.from_env("LLM_MODEL", max_tokens=1024)
    judge = None if args.no_judge else ChatClient.from_env("JUDGE_MODEL", max_tokens=300)
    metrics = answer_run(
        Path(args.results), generator, judge, args.max_pages, args.token_budget, args.form,
        args.regions, args.limit, args.review_n,
    )  # fmt: skip
    print(metrics)


if __name__ == "__main__":
    main()
