"""Stage 4 end to end on the tiny dataset: retrieve, rerank, answer, verify, analyze."""

import json

import pytest

from helpers import FakeChat, FakeLateEncoder, HashingEmbedder, answer_json
from mmrag.experiments.answer import answer_run, page_forms
from mmrag.experiments.common import read_json, read_jsonl, write_json
from mmrag.experiments.failure_analysis import analyze_failures, to_markdown
from mmrag.experiments.ladder import ladder_table
from mmrag.experiments.run import build_parser, run_experiment

HYBRID = ["--pipeline", "hybrid", "--min-text-chars", "20", "--no-analyzer-images"]


@pytest.fixture
def run_dir(dataset_dir, tmp_path, monkeypatch):
    import mmrag.experiments.run as run_module
    from mmrag.experiments import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    def make(name, *extra, **overrides):
        argv = ["--dataset", str(dataset_dir), "--n-queries", "0", "--date", "2000-01-01"]
        argv += ["--work-dir", str(tmp_path / "work" / name), "--name", name, *extra]
        overrides.setdefault("embedder", HashingEmbedder())
        overrides.setdefault("visual_encoder", FakeLateEncoder())
        return run_experiment(build_parser().parse_args(argv), **overrides)

    return make


def generator():
    """Answers each question from its text, citing the first page it was shown."""

    def reply(system, parts):
        pages = [p["text"] for p in parts if p["type"] == "text" and p["text"].startswith("[PAGE")]
        first = pages[0].split("]")[0].removeprefix("[PAGE ").strip()
        question = parts[-1]["text"]
        if "parental" in question:
            return answer_json("Sixteen weeks.", [(first, "sixteen weeks of paid parental leave")])
        if "salary" in question:
            return answer_json("Every April.", [(first, "reviewed every quarter by finance")])
        if "TX-900" in question:
            return answer_json("The travel desk.", [("999", "travel desk")])
        return answer_json("EMEA grew fastest.", [(first, "")])

    return FakeChat(reply)


def judge():
    def reply(system, prompt):
        verdict = "incorrect" if "TX-900" in prompt else "correct"
        return json.dumps({"verdict": verdict, "reason": "test"})

    return FakeChat(reply)


def test_rerank_run_records_reranker_use(run_dir, tmp_path):
    from helpers import PAGES, VISIBLE
    from mmrag.visual.store import MultiVectorStore

    encoder = FakeLateEncoder()
    pages = MultiVectorStore.from_embeddings(
        list(PAGES), [encoder._vectors(VISIBLE[p]) for p in PAGES], "float16"
    )
    out = run_dir(
        "rerank", *HYBRID, "--rerank", "maxsim", "--rerank-top-n", "4",
        visual_encoder=encoder, visual_pages=pages,
    )  # fmt: skip
    assert read_json(out / "metrics.json")["recall_10"] == 1.0
    cost = read_json(out / "cost.json")
    assert cost["after_search"]["rerank"]["reranker"] == "maxsim"
    assert cost["after_search"]["rerank"]["pages_scored"] > 0
    assert "rerank_seconds" in cost["after_search"]["timings"]
    # With every page's vectors available, MaxSim puts the chart page first.
    chart_ranking = read_json(out / "run.json")["14"]
    assert max(chart_ranking, key=chart_ranking.get) == "3"


def test_llm_rerank_uses_the_injected_client(run_dir):
    client = FakeChat(lambda system, parts: json.dumps({"ranking": []}))
    out = run_dir("llm", *HYBRID, "--rerank", "llm", chat_client=client)
    assert client.calls == 4  # one listwise call per query
    assert read_json(out / "cost.json")["after_search"]["rerank"]["unparsed_replies"] == 4


def test_answers_are_generated_verified_and_graded(run_dir):
    out = run_dir("hybrid", *HYBRID)
    gen, grader = generator(), judge()
    metrics = answer_run(out, gen, grader, max_pages=3, review_n=3)

    records = {r["query_id"]: r for r in read_jsonl(out / "answers.jsonl")}
    assert set(records) == {"10", "11", "12", "14"}
    assert gen.calls == 4 and grader.calls == 4

    # Verified quote on the right page.
    assert records["10"]["citations"]["verified"] == 1
    assert records["10"]["provenance"]["cites_a_relevant_page"] == 1.0
    # Invented quote: the page is right, the quote is not on it.
    assert records["11"]["citations"]["quote_not_found"] == 1
    # Cites a page that was never shown: rejected, and it earns no provenance credit.
    assert records["12"]["citations"]["page_not_in_context"] == 1
    assert records["12"]["provenance"]["cited_pages"] == 0.0
    assert records["12"]["judge_verdict"] == "incorrect"

    assert metrics["answer"]["mean_score"] == 0.75
    assert metrics["citations"]["n_citations"] == 4
    assert metrics["citations"]["answers_all_citations_ok"] == 0.5
    assert metrics["context_has_relevant_page"] == 1.0

    # Visual pages were shown as images, text pages as text.
    forms = {c["page"]: c["form"] for r in records.values() for c in r["context"]}
    assert forms["3"] == "image" and forms["0"] == "text"
    cost = read_json(out / "answer_cost.json")
    assert cost["generator"]["calls"] == 4 and cost["images_per_query"] > 0
    assert (out / "review_sample.csv").exists()


def test_generation_without_judge_and_with_limit(run_dir):
    out = run_dir("text_only", "--pipeline", "text")
    gen = generator()
    metrics = answer_run(out, gen, judge=None, limit=2)
    assert gen.calls == 2 and metrics["answer"] is None
    assert not (out / "review_sample.csv").exists()
    assert all(c["form"] == "text" for r in read_jsonl(out / "answers.jsonl") for c in r["context"])


def test_page_forms_follow_pipeline_and_override(tmp_path):
    ids = ["a", "b"]
    assert page_forms(tmp_path, "visual", "auto", ids) == {"a": "image", "b": "image"}
    assert page_forms(tmp_path, "text", "auto", ids) == {"a": "text", "b": "text"}
    (tmp_path / "page_routes.json").write_text('{"a": "text", "b": "both"}', encoding="utf-8")
    assert page_forms(tmp_path, "hybrid", "auto", ids) == {"a": "text", "b": "image"}
    assert page_forms(tmp_path, "hybrid", "text", ids) == {"a": "text", "b": "text"}


def test_failure_analysis_names_the_stage_that_lost_each_page(run_dir):
    full = run_dir("hybrid", *HYBRID)
    report = analyze_failures(full)
    assert report["summary"]["n_missed_pages"] == 0
    assert report["summary"]["page_recall_by_page_route"]["visual"]["recall"] == 1.0
    assert report["summary"]["relevant_pages_found_by_one_route_only"]["visual"] >= 1

    ablated = run_dir("no_visual", *HYBRID, "--routes", "text,bm25")
    report = analyze_failures(ablated)
    summary = report["summary"]
    assert summary["n_missed_pages"] == 1
    assert summary["misses_by_stage"] == {"not_retrieved_by_any_route": 1}
    miss = report["misses"][0]
    assert (miss["query_id"], miss["page"], miss["page_route"]) == ("14", "3", "visual")
    assert report["queries"][0]["query_id"] == "14"  # worst query first

    single = analyze_failures(run_dir("text_only", "--pipeline", "text"))
    assert single["summary"]["misses_by_stage"] == {"not_in_top_k": 1}

    markdown = to_markdown(report)
    assert "not_retrieved_by_any_route | 1" in markdown and "`14` found 0/1" in markdown


def test_failure_analysis_splits_wrong_answers_by_cause(run_dir):
    out = run_dir("no_visual", *HYBRID, "--routes", "text,bm25")

    def grade(system, prompt):
        wrong = "TX-900" in prompt or "headcount" in prompt
        return json.dumps({"verdict": "incorrect" if wrong else "correct", "reason": ""})

    answer_run(out, generator(), FakeChat(grade))
    answers = analyze_failures(out)["summary"]["answers"]
    assert answers["n_not_correct"] == 2
    assert answers["causes"] == {
        "retrieval: no relevant page shown": 1,  # the chart question
        "generation: relevant page shown but not cited": 1,  # cited a page it never saw
    }
    assert answers["answers_with_failed_citation_checks"] == 2


def test_ladder_table_decides_from_stored_numbers(run_dir, tmp_path):
    run_dir("text_only", "--pipeline", "text")
    hybrid = run_dir("hybrid", *HYBRID)
    answer_run(hybrid, generator(), judge())
    steps = [
        {"step": "0", "change": "baseline", "run": "2000-01-01_text_only", "kind": "baseline"},
        {"step": "1", "change": "+ routes", "run": "2000-01-01_hybrid",
         "against": "2000-01-01_text_only", "kind": "quality", "metric": "ndcg"},
        {"step": "2", "change": "same again", "run": "2000-01-01_hybrid",
         "against": "2000-01-01_hybrid", "kind": "efficiency", "metric": "ndcg"},
    ]  # fmt: skip
    write_json(tmp_path / "results" / "ladder.json", steps)

    lines = ladder_table(tmp_path / "results").splitlines()
    baseline, quality, efficiency = (line.split(" | ") for line in lines[2:])
    assert baseline[4] == "-" and quality[4] == "0.750"  # answer score only where it was run
    # Four queries cannot show a significant gain, so the quality step is not kept;
    # an efficiency step with no loss is.
    assert quality[-2] == "no" and efficiency[-2] == "yes"
    assert "+0.000 [+0.000, +0.000] ndcg" in efficiency[7]

    # A held-out table uses the same code; an "ablation" row is tested but not decided.
    write_json(tmp_path / "results" / "heldout.json", [{**steps[1], "kind": "ablation"}])
    ablation = ladder_table(tmp_path / "results", "heldout").splitlines()[2].split(" | ")
    assert ablation[-2] == "-" and "ndcg" in ablation[7]


def test_breakdown_table_reads_stored_per_query_scores(run_dir, tmp_path):
    from mmrag.experiments.breakdown import breakdown_table

    run_dir("text_only", "--pipeline", "text")
    answer_run(run_dir("hybrid", *HYBRID), generator(), judge())
    runs = {"baseline": "2000-01-01_text_only", "final": "2000-01-01_hybrid"}
    write_json(tmp_path / "results" / "breakdown.json", {"tiny": runs})

    lines = [line.split(" | ") for line in breakdown_table(tmp_path / "results").splitlines()]
    assert lines[0][3:] == [
        "NDCG@10 baseline",
        "NDCG@10 final",
        "answer baseline",
        "answer final |",
    ]
    everything = lines[2]
    assert everything[1:3] == ["all", "4"]
    # The baseline was not judged; the judged run's mean is the one the ladder shows.
    assert everything[5] == "-" and everything[6] == "0.750 |"
    assert [row[1] for row in lines[-4:]] == [
        "1 relevant", "2-4 relevant", "5-9 relevant", "10+ relevant",
    ]  # fmt: skip
