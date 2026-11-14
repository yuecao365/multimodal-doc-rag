"""The experiment runner end to end on a tiny local dataset, with offline models."""

import json

import pytest

from helpers import HashingEmbedder
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
        argv = ["--dataset", str(dataset_dir), "--n-queries", "3", "--date", "2000-01-01"]
        return build_parser().parse_args([*argv, "--work-dir", str(tmp_path / "work"), *extra])

    return make


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_text_pipeline_writes_reproducible_results(run_args, tmp_path):
    out = run_experiment(run_args("--pipeline", "text", "--name", "t"), embedder=HashingEmbedder())

    assert out == tmp_path / "results" / "2000-01-01_t"
    metrics = read(out / "metrics.json")
    assert metrics["n_queries"] == 3 and metrics["n_corpus_pages"] == 6
    assert metrics["n_pages_empty_text"] == 1
    assert metrics["recall_10"] == 1.0
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
