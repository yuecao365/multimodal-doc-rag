"""The experiment runner end to end on a tiny local dataset, with offline models."""

import json

import pytest

from helpers import FakeLateEncoder, HashingEmbedder
from mmrag.adapters.vidore_pipeline import MMRAGPipeline
from mmrag.data.vidore import load_vidore_subset
from mmrag.experiments.run import build_parser, run_experiment


@pytest.fixture
def run_args(dataset_dir, tmp_path, monkeypatch):
    """Build runner arguments that keep every output under tmp_path."""
    import mmrag.experiments.run as run_module
    from mmrag.experiments import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(run_module, "ROOT", tmp_path)

    def make(*extra):
        argv = ["--dataset", str(dataset_dir), "--n-queries", "0", "--date", "2000-01-01"]
        return build_parser().parse_args([*argv, "--work-dir", str(tmp_path / "work"), *extra])

    return make


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_text_pipeline_writes_reproducible_results(run_args, tmp_path):
    out = run_experiment(run_args("--pipeline", "text", "--name", "t"), embedder=HashingEmbedder())

    assert out == tmp_path / "results" / "2000-01-01_t"
    metrics = read(out / "metrics.json")
    assert metrics["n_queries"] == 4 and metrics["n_corpus_pages"] == 6
    assert metrics["n_pages_empty_text"] == 1
    # Three questions are answered by text pages; the fourth needs the chart page,
    # which has no extracted text and so cannot be found by the text route.
    assert metrics["recall_10"] == 0.75
    config = read(out / "config.json")
    assert config["pipeline"]["pipeline"] == "text" and config["seed"] == 0
    assert read(out / "cost.json")["index_storage_bytes"] > 0
    for name in ("environment.json", "queries.json", "run.json", "per_query.jsonl"):
        assert (out / name).exists()


def test_leaderboard_adapter_has_basepipeline_shape(dataset_dir, tmp_path):
    data = load_vidore_subset(str(dataset_dir), n_queries=None, language="english")
    pipeline = MMRAGPipeline(
        pipeline="text", work_dir=str(tmp_path / "lb"), embedder=HashingEmbedder()
    )
    pipeline.index(data.corpus_ids, None, data.corpus_texts, dataset_name="org/tiny")
    run, infos = pipeline.search(data.query_ids, data.queries)

    assert set(run) == set(data.query_ids)
    assert all(isinstance(s, float) for scores in run.values() for s in scores.values())
    assert max(run["10"], key=run["10"].get) == "0"
    assert infos["index_storage_bytes"] > 0
    pipeline._pipeline.close()


def test_visual_pipeline_finds_the_chart_page(run_args):
    out = run_experiment(
        run_args("--pipeline", "visual", "--quantization", "int8"),
        visual_encoder=FakeLateEncoder(),
    )
    assert out.name == "2000-01-01_visual"
    assert read(out / "metrics.json")["recall_10"] == 1.0
    cost = read(out / "cost.json")
    assert cost["after_index"]["visual_index"]["quantization"] == "int8"
    assert cost["after_index"]["pages_encoded_now"] == 6
    assert cost["index_storage_bytes_per_page"] > 0


def test_encode_then_retrieve_from_files(dataset_dir, tmp_path, run_args):
    """The two-environment flow: encode the subset to files, retrieve from the saved vectors."""
    from mmrag.visual.encode_cli import encode_dataset

    visual_dir = tmp_path / "vectors"
    encode_dataset(FakeLateEncoder(), str(dataset_dir), visual_dir, model="fake")
    out = run_experiment(run_args("--pipeline", "visual", "--visual-store", str(visual_dir)))
    assert read(out / "metrics.json")["recall_10"] == 1.0
    assert read(out / "cost.json")["after_index"]["pages_encoded_now"] == 0


def test_bm25_pipeline(run_args):
    out = run_experiment(run_args("--pipeline", "bm25"))
    assert read(out / "metrics.json")["recall_10"] == 0.75  # same blind spot as the text route


def test_hybrid_pipeline_routes_pages_and_records_every_route(run_args):
    out = run_experiment(
        # The toy pages are a sentence long and their images are synthetic, so route on
        # text alone with a threshold that fits them.
        run_args("--pipeline", "hybrid", "--min-text-chars", "20", "--no-analyzer-images"),
        embedder=HashingEmbedder(),
        visual_encoder=FakeLateEncoder(),
    )
    assert read(out / "metrics.json")["recall_10"] == 1.0
    routes = read(out / "page_routes.json")
    assert routes["3"] == "visual" and routes["0"] == "text"
    for name in ("text", "bm25", "visual", "fused"):
        assert (out / f"run_{name}.json").exists()
    cost = read(out / "cost.json")
    summary = cost["after_index"]["routing_summary"]
    assert summary["n_text"] + summary["n_visual"] == 6
    # Only the visual pages were encoded, not the whole corpus.
    assert cost["after_index"]["pages_encoded_now"] == summary["n_visual"] < 6
    assert set(cost["after_search"]["storage_bytes_by_route"]) == {"text", "bm25", "visual"}
    assert "route_runs" not in cost["after_search"]


def test_hybrid_ablation_without_visual_route(run_args):
    out = run_experiment(
        run_args(
            "--pipeline",
            "hybrid",
            "--routes",
            "text,bm25",
            "--name",
            "no_visual",
            "--min-text-chars",
            "20",
            "--no-analyzer-images",
        ),
        embedder=HashingEmbedder(),
    )
    assert out.name == "2000-01-01_no_visual"
    assert read(out / "metrics.json")["recall_10"] == 0.75
    assert read(out / "config.json")["pipeline"]["routes"] == "text,bm25"
