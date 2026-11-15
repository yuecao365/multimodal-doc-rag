"""Generate and grade answers for a finished retrieval run.

    uv run python -m mmrag.experiments.answer --results results/2026-11-14_hr_baseline

Reads the run's rankings, shows the top pages to the VLM (as images or text),
checks the citations in code, grades the answer against the dataset's reference
answer, and writes into the same results directory:

    answers.jsonl          one record per query (answer, citations, checks, grade)
    answer_metrics.json    answer quality, citation validity, provenance
    answer_cost.json       model calls, tokens, cost, latency
    review_sample.csv      a sample for a person to label, to check the grader

This calls the chat models configured in .env (LLM_MODEL, JUDGE_MODEL). It costs money.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mmrag.data.vidore import VidoreSubset, load_vidore_subset
from mmrag.eval.answers import export_review_sample, judge_answer, summarize_judgments
from mmrag.eval.breakdown import breakdown
from mmrag.eval.provenance import citation_page_metrics, mean_of, region_iou
from mmrag.eval.significance import mean_ci
from mmrag.experiments import spend
from mmrag.experiments.common import ROOT, read_json, write_json, write_jsonl
from mmrag.generation.answer import generate_answer
from mmrag.generation.citations import summarize_reports, verify_citations
from mmrag.generation.context import assemble_context
from mmrag.retrieval.fusion import ranked_ids

logger = logging.getLogger("mmrag.answer")

# Changes the texts and images of the pages about to be shown: (texts, images) -> same.
PageEdit = Callable[[dict[str, str], dict[str, Any]], tuple[dict[str, str], dict[str, Any]]]


def page_forms(results: Path, pipeline: str, form: str, page_ids: list[str]) -> dict[str, str]:
    """How each page is shown to the model.

    "auto" follows the page analyzer: pages it judged visual are shown as images,
    text pages as text; a visual-only run shows images, a text-only run shows text.
    """
    if form in ("image", "text"):
        return dict.fromkeys(page_ids, form)
    routes_file = results / "page_routes.json"
    if routes_file.exists():
        routes = read_json(routes_file)
        return {p: "text" if routes.get(p) == "text" else "image" for p in page_ids}
    default = "image" if pipeline == "visual" else "text"
    return dict.fromkeys(page_ids, default)


@dataclass
class RetrievalRun:
    """A finished retrieval run and the dataset it was made on."""

    data: VidoreSubset
    run: dict[str, dict[str, float]]
    forms: dict[str, str]
    selected: dict[str, list[str]]  # pages an agent chose to answer from, if any
    texts: dict[str, str] = field(init=False)
    labels: dict[str, str] = field(init=False)
    _images_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        data = self.data
        self.texts = dict(zip(data.corpus_ids, data.corpus_texts))
        self.labels = {
            cid: f"{doc} p.{page + 1}"
            for cid, doc, page in zip(data.corpus_ids, data.doc_ids, data.page_numbers)
        }
        self.query_of = dict(zip(data.query_ids, data.queries))

    @classmethod
    def load(cls, results: Path, form: str = "auto") -> RetrievalRun:
        config = read_json(results / "config.json")
        data = load_vidore_subset(
            config["dataset"], config["n_queries"], config["language"], config["seed"]
        )
        forms = page_forms(results, config["pipeline"]["pipeline"], form, data.corpus_ids)
        selected_file = results / "selected_pages.json"
        selected = read_json(selected_file) if selected_file.exists() else {}
        return cls(data, read_json(results / "run.json"), forms, selected)

    def pages(self, qid: str, max_pages: int) -> list[str]:
        """The pages to answer from: the agent's selection, else the top of the ranking."""
        return (self.selected.get(qid) or ranked_ids(self.run.get(qid, {})))[:max_pages]

    def answer(
        self,
        generator: Any,
        qid: str,
        pages: list[str],
        judge: Any | None = None,
        token_budget: int = 12_000,
        with_regions: bool = False,
        defended: bool = False,
        page_edit: PageEdit | None = None,
    ) -> dict:
        """Answer one question from ``pages`` and check the answer; returns the record."""
        data, query = self.data, self.query_of[qid]
        need_images = [p for p in pages if self.forms[p] == "image" or not self.texts[p].strip()]
        with self._images_lock:  # one parquet reader at a time
            images = data.load_page_images(need_images) if need_images else {}
        texts = self.texts
        if page_edit is not None:
            texts, images = page_edit({p: texts[p] for p in pages}, images)
        contexts = assemble_context(
            pages, texts, images, self.forms, max_pages=len(pages), token_budget=token_budget,
            labels=self.labels,
        )  # fmt: skip

        t0 = time.perf_counter()
        answer = generate_answer(generator, query, contexts, with_regions, defended)
        seconds = time.perf_counter() - t0

        shown = [c.page_id for c in contexts]
        report = verify_citations(answer, shown, self.texts)
        regions = []
        if with_regions:
            for citation in answer.citations:
                gold = data.bboxes.get(qid, {}).get(citation.page)
                if citation.region and gold and citation.page in images:
                    score = region_iou(citation.region, gold, images[citation.page].size)
                    regions.append({"page": citation.page, "iou": round(score, 3)})

        reference = data.answers.get(qid, "")
        judgment = None
        if judge is not None and reference:
            judgment = judge_answer(judge, query, reference, answer.answer)
        relevant = {p for p, score in data.qrels[qid].items() if score > 0}
        return {
            "query_id": qid,
            "question": query,
            "reference": reference,
            "answer": answer.answer,
            "answer_parsed": answer.parsed,
            "abstained": not answer.answerable,
            "generate_seconds": round(seconds, 2),
            "context": [
                {"page": c.page_id, "form": c.form, "tokens": c.tokens, "truncated": c.truncated}
                for c in contexts
            ],
            "relevant_pages_in_context": sorted(set(shown) & relevant),
            "citations": report.to_dict(),
            "provenance": citation_page_metrics(report.valid_pages, data.qrels[qid]),
            "regions": regions,
            "judge_verdict": judgment["verdict"] if judgment else None,
            "judge_score": judgment["score"] if judgment else None,
            "judge_reason": judgment["reason"] if judgment else None,
        }


def answer_metrics(records: list[dict], data: VidoreSubset, judged: bool) -> dict:
    n = max(1, len(records))
    scores = {r["query_id"]: r["judge_score"] for r in records if r["judge_score"] is not None}
    region_scores = [r["iou"] for record in records for r in record["regions"]]
    judgments = [
        {"verdict": r["judge_verdict"], "score": r["judge_score"]}
        for r in records
        if r["reference"]
    ]
    return {
        "n_queries": len(records),
        "answers_unparsed": sum(1 for r in records if not r["answer_parsed"]),
        "abstained": sum(1 for r in records if r["abstained"]) / n,
        "answer": summarize_judgments(judgments) if judged else None,
        "answer_score_ci95": mean_ci(list(scores.values()))["ci95"] if scores else None,
        "answer_score_breakdown": breakdown(scores, data) if scores else None,
        "citations": summarize_reports([r["citations"] for r in records]),
        "provenance": mean_of([r["provenance"] for r in records]),
        "pages_in_context": sum(len(r["context"]) for r in records) / n,
        "context_has_relevant_page": sum(1 for r in records if r["relevant_pages_in_context"]) / n,
        "region": {
            "n_scored": len(region_scores),
            "mean_iou": sum(region_scores) / len(region_scores) if region_scores else None,
        },
    }


def answer_run(
    results: Path,
    generator: Any,
    judge: Any | None = None,
    max_pages: int = 5,
    token_budget: int = 12_000,
    form: str = "auto",
    with_regions: bool = False,
    limit: int | None = None,
    workers: int = 8,
    review_n: int = 30,
) -> dict:
    spend.check_budget()
    retrieval = RetrievalRun.load(results, form)
    data = retrieval.data
    query_ids = data.query_ids[:limit] if limit else data.query_ids

    def answer_one(qid: str) -> dict:
        pages = retrieval.pages(qid, max_pages)
        return retrieval.answer(generator, qid, pages, judge, token_budget, with_regions)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(answer_one, query_ids))

    write_jsonl(results / "answers.jsonl", records)
    n = max(1, len(records))
    metrics = {
        "settings": {
            "max_pages": max_pages,
            "token_budget": token_budget,
            "form": form,
            "with_regions": with_regions,
        },
        **answer_metrics(records, data, judged=judge is not None),
    }
    write_json(results / "answer_metrics.json", metrics)
    usage = {
        "generator": generator.usage() if hasattr(generator, "usage") else None,
        "judge": judge.usage() if judge is not None and hasattr(judge, "usage") else None,
    }
    write_json(
        results / "answer_cost.json",
        {
            "generate_seconds_per_query": round(sum(r["generate_seconds"] for r in records) / n, 3),
            "generate_cost_usd_per_query": round(spend.total_cost(usage["generator"]) / n, 6),
            "context_tokens_estimated_per_query": round(
                sum(c["tokens"] for r in records for c in r["context"]) / n
            ),
            "images_per_query": sum(
                1 for r in records for c in r["context"] if c["form"] == "image"
            )
            / n,
            **usage,
        },
    )
    spend.record(results.name, f"answers ({len(records)} queries)", usage)
    if judge is not None:
        export_review_sample(records, results / "review_sample.csv", n=review_n)
    return metrics


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
    p.add_argument("--workers", type=int, default=8, help="questions answered in parallel")
    args = p.parse_args(argv)

    from mmrag.models import ChatClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    load_dotenv(ROOT / ".env")
    generator = ChatClient.from_env("LLM_MODEL")
    judge = None if args.no_judge else ChatClient.from_env("JUDGE_MODEL")
    metrics = answer_run(
        Path(args.results), generator, judge, args.max_pages, args.token_budget, args.form,
        args.regions, args.limit, args.workers,
    )  # fmt: skip
    print({k: v for k, v in metrics.items() if k != "answer_score_breakdown"})


if __name__ == "__main__":
    main()
