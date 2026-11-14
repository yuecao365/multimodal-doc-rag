import csv

from helpers import FakeChat
from mmrag.eval.answers import (
    export_review_sample,
    judge_agreement,
    judge_answer,
    summarize_judgments,
)


def test_judge_parses_verdict_and_scores():
    client = FakeChat(['{"verdict": "partially_correct", "reason": "misses the year"}'])
    result = judge_answer(client, "q", "ref", "cand")
    assert result == {"verdict": "partially_correct", "score": 0.5, "reason": "misses the year"}
    _, prompt = client.prompts[0]
    assert "Reference answer:\nref" in prompt and "Answer to grade:\ncand" in prompt


def test_judge_reports_unparseable_reply_instead_of_guessing():
    result = judge_answer(FakeChat(["looks fine to me"]), "q", "ref", "cand")
    assert result["verdict"] is None and result["score"] is None


def test_summary_excludes_unparsed_from_the_mean():
    judgments = [
        {"verdict": "correct", "score": 1.0},
        {"verdict": "incorrect", "score": 0.0},
        {"verdict": None, "score": None},
    ]
    summary = summarize_judgments(judgments)
    assert summary["n_judged"] == 2 and summary["n_unparsed"] == 1
    assert summary["mean_score"] == 0.5 and summary["accuracy_strict"] == 0.5


def test_review_sample_roundtrip_and_agreement(tmp_path):
    records = [
        {
            "query_id": str(i),
            "question": "q",
            "reference": "r",
            "answer": "a",
            "judge_verdict": "correct" if i % 2 else "incorrect",
        }
        for i in range(10)
    ]
    path = tmp_path / "review.csv"
    assert export_review_sample(records, path, n=4, seed=0) == 4
    assert judge_agreement(path)["n"] == 0  # nothing labelled yet

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # The human agrees on all but the first row.
    flip = {"correct": "incorrect", "incorrect": "correct"}
    for i, row in enumerate(rows):
        verdict = row["judge_verdict"]
        row["human_verdict"] = flip[verdict] if i == 0 else verdict
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = judge_agreement(path)
    assert result["n"] == 4 and result["agreement"] == 0.75
    assert -1.0 <= result["cohen_kappa"] < 1.0
