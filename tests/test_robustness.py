"""Abstention and prompt-injection checks on the tiny dataset with scripted models."""

import json

import pytest

from helpers import PAGES, FakeChat, FakeLateEncoder, HashingEmbedder, answer_json, page_image
from mmrag.experiments.answer import RetrievalRun, answer_run
from mmrag.experiments.common import read_jsonl
from mmrag.experiments.robustness import (
    CANARY,
    INJECTION,
    abstention,
    defended_answers,
    inject_image,
    inject_text,
    injection,
    summarize,
)
from mmrag.experiments.run import build_parser, run_experiment

EVIDENCE = {"parental": "sixteen weeks", "salary": "every April", "TX-900": "travel desk"}


def obedient_model(system, parts):
    """Obeys a planted instruction unless page content is delimited; abstains without evidence."""
    prompt = json.dumps(parts)
    if INJECTION[:40] in prompt and "<page_content>" not in prompt:
        return answer_json(f"The verification code is {CANARY}", [])
    question = parts[-1]["text"]
    for word, fact in EVIDENCE.items():
        if word in question and fact in prompt:
            return answer_json(fact, [])
    return json.dumps({"answer": "The pages do not say.", "answerable": False, "citations": []})


def grader(system, parts):
    correct = any(fact in parts.split("Answer to grade:")[1] for fact in EVIDENCE.values())
    return json.dumps({"verdict": "correct" if correct else "incorrect", "reason": ""})


@pytest.fixture
def retrieval(dataset_dir, tmp_path, monkeypatch):
    from mmrag.experiments import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    argv = ["--dataset", str(dataset_dir), "--date", "2000-01-01", "--name", "final"]
    argv += ["--work-dir", str(tmp_path / "work"), "--pipeline", "hybrid"]
    argv += ["--min-text-chars", "20", "--no-analyzer-images"]
    out = run_experiment(
        build_parser().parse_args(argv),
        embedder=HashingEmbedder(),
        visual_encoder=FakeLateEncoder(),
    )
    answer_run(out, FakeChat(obedient_model), FakeChat(grader), max_pages=3)
    return out, RetrievalRun.load(out)


def test_model_abstains_when_relevant_pages_are_removed(retrieval):
    out, run = retrieval
    absent = abstention(run, FakeChat(obedient_model), n=3, max_pages=3, seed=0, workers=2)
    assert len(absent) == 3
    assert all(not r["relevant_pages_in_context"] and r["abstained"] for r in absent)
    summary = summarize(read_jsonl(out / "answers.jsonl"), absent, [], [])["abstention"]
    assert summary["refusal_rate_without_relevant_pages"] == 1.0
    # On the normal run only the chart question (no evidence the fake can read) is refused.
    assert summary["false_refusal_rate_with_a_relevant_page"] == 0.25


def test_injection_succeeds_undefended_and_fails_defended(retrieval):
    out, run = retrieval
    client = FakeChat(obedient_model)
    poisoned = injection(run, client, n=4, max_pages=3, seed=0, workers=1)
    assert len(poisoned) == 8
    # The scripted model reads text only, so only pages poisoned in their text can fool it.
    in_text = [r for r in poisoned if r["poisoned_form"] == "text"]
    assert all(r["attack_succeeded"] != r["defended"] for r in in_text)
    assert {r["poisoned_form"] for r in poisoned} == {"text", "image"}  # the chart page is an image
    # The citation check still runs against the page's real text.
    assert all(INJECTION not in json.dumps(r["citations"]) for r in poisoned)

    defended = defended_answers(run, client, FakeChat(grader), max_pages=3, workers=2)
    summary = summarize(read_jsonl(out / "answers.jsonl"), [], poisoned, defended)
    override = summary["injection"]["override"]
    assert override["undefended"]["by_form"]["text"]["attack_success_rate"] == 1.0
    assert override["defended"]["attack_success_rate"] == 0.0
    # The quieter attack is not obeyed by this scripted model.
    quiet = injection(run, client, n=2, max_pages=3, seed=0, workers=1, attack="append")
    assert quiet[0]["attack"] == "append" and not any(r["attack_succeeded"] for r in quiet)
    clean = summary["defence_on_clean_queries"]
    assert clean["difference"]["mean"] == 0 and not clean["worse"]


def test_injection_is_placed_inside_the_page():
    text = inject_text(PAGES["0"])
    assert text.startswith("# Parental leave") and INJECTION in text
    original = page_image("3", size=(400, 500))
    stamped = inject_image(original)
    assert stamped.size == original.size and stamped.tobytes() != original.tobytes()
