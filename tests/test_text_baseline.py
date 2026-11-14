"""Offline end-to-end check of the text route against real RAG-Anything / LightRAG."""

import json

from helpers import HashingEmbedder
from mmrag.eval.retrieval import evaluate_run
from mmrag.models import NoKGLLM
from mmrag.pipelines.text_baseline import TextBaselinePipeline

PAGES = {
    "p-leave": "# Parental leave\nEmployees receive sixteen weeks of paid parental leave.",
    "p-salary": "# Salary bands\nEngineering salary bands are reviewed every April.",
    "p-remote": "# Remote work\nRemote work requires manager approval and a home office audit.",
    "p-blank": "   ",
}


def test_indexes_pages_without_kg_and_retrieves_the_right_page(tmp_path, monkeypatch):
    # The baseline must work on a machine without MinerU.
    from raganything.parser import MineruParser

    monkeypatch.setattr(MineruParser, "check_installation", lambda self: False)
    llm = NoKGLLM()
    pipeline = TextBaselinePipeline(
        working_dir=str(tmp_path / "store"),
        embedder=HashingEmbedder(),
        embedding_dim=HashingEmbedder.dim,
        llm=llm,
    )
    pipeline.index(list(PAGES), None, list(PAGES.values()))
    run = pipeline.search(
        ["q1", "q2"], ["how many weeks of paid parental leave", "when are salary bands reviewed"]
    )
    pipeline.close()

    assert pipeline.skipped_empty == ["p-blank"]
    assert max(run["q1"], key=run["q1"].get) == "p-leave"
    assert max(run["q2"], key=run["q2"].get) == "p-salary"
    assert set(run["q1"]) <= set(PAGES) - {"p-blank"}
    assert llm.calls > 0  # entity extraction was invoked, and answered with "nothing"

    mean, _ = evaluate_run({"q1": {"p-leave": 1}, "q2": {"p-salary": 1}}, run)
    assert mean["recall_10"] == 1.0

    # No knowledge graph was built: the entity vector store stays empty.
    entity_file = next((tmp_path / "store").rglob("vdb_entities.json"))
    entities = json.loads(entity_file.read_text(encoding="utf-8"))
    assert entities["data"] == []
    assert pipeline.storage_bytes() > 0
