"""Answer scoring: an LLM grader against the reference answer, plus tools to
check the grader itself against a human-labelled sample."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

VERDICT_SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = """You grade an answer to a question about a document against a reference answer.

Verdicts:
- "correct": states the same key facts as the reference; wording and extra correct detail do not matter.
- "partially_correct": some key facts of the reference are present, others are missing or wrong.
- "incorrect": contradicts the reference, misses its key facts, or declines to answer.

Judge only factual agreement with the reference. Ignore citations and style.
Reply with one JSON object and nothing else:
{"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one sentence>"}"""


def judge_answer(client: Any, question: str, reference: str, candidate: str) -> dict:
    """Grade one answer. Returns {"verdict", "score", "reason"}; verdict None if unparseable."""
    prompt = (
        f"Question:\n{question}\n\nReference answer:\n{reference}\n\nAnswer to grade:\n{candidate}"
    )
    raw = client.complete(JUDGE_SYSTEM, prompt, json_mode=True)
    verdict, reason = None, ""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = str(obj.get("verdict", "")).strip().lower().replace(" ", "_")
            reason = str(obj.get("reason", ""))
        except json.JSONDecodeError:
            pass
    if verdict not in VERDICT_SCORES:
        return {"verdict": None, "score": None, "reason": raw.strip()[:300]}
    return {"verdict": verdict, "score": VERDICT_SCORES[verdict], "reason": reason}


def summarize_judgments(judgments: list[dict]) -> dict:
    scored = [j for j in judgments if j.get("score") is not None]
    n = len(scored)
    counts = {v: sum(1 for j in scored if j["verdict"] == v) for v in VERDICT_SCORES}
    return {
        "n_judged": n,
        "n_unparsed": len(judgments) - n,
        "mean_score": sum(j["score"] for j in scored) / n if n else None,
        "accuracy_strict": counts["correct"] / n if n else None,
        **{f"n_{v}": c for v, c in counts.items()},
    }


# -- human spot check ---------------------------------------------------------

REVIEW_COLUMNS = ["query_id", "question", "reference", "answer", "judge_verdict", "human_verdict"]


def export_review_sample(records: list[dict], path: str | Path, n: int = 20, seed: int = 0) -> int:
    """Write a CSV of ``n`` graded answers for a person to label.

    The reviewer fills ``human_verdict`` with correct / partially_correct / incorrect.
    """
    records = [r for r in records if r.get("judge_verdict")]
    sample = random.Random(seed).sample(records, min(n, len(records)))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(sample, key=lambda r: str(r["query_id"])):
            writer.writerow({**r, "human_verdict": ""})
    return len(sample)


def judge_agreement(path: str | Path) -> dict:
    """Agreement between grader and human on a filled-in review CSV (raw and Cohen's kappa)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("human_verdict") or "").strip()]
    pairs = [
        (r["judge_verdict"].strip().lower(), r["human_verdict"].strip().lower().replace(" ", "_"))
        for r in rows
    ]
    pairs = [(j, h) for j, h in pairs if h in VERDICT_SCORES]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "cohen_kappa": None}
    observed = sum(1 for j, h in pairs if j == h) / n
    expected = sum(
        (sum(1 for j, _ in pairs if j == v) / n) * (sum(1 for _, h in pairs if h == v) / n)
        for v in VERDICT_SCORES
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": n, "agreement": observed, "cohen_kappa": kappa}
