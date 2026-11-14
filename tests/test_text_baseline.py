"""Offline end-to-end check of the baseline against real RAG-Anything / LightRAG.

Uses a deterministic bag-of-words embedder, so no network or API key is needed.
"""

import hashlib
import json
import re

import numpy as np

from mmrag.eval.retrieval import evaluate_run
from mmrag.models import NoKGLLM
from mmrag.pipelines.text_baseline import TextBaselinePipeline

DIM = 256


class HashingEmbedder:
    async def __call__(self, texts, **kwargs):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


PAGES = {
    "p-leave": "# Parental leave\nEmployees receive sixteen weeks of paid parental leave.",
    "p-salary": "# Salary bands\nEngineering salary bands are reviewed every April.",
    "p-remote": "# Remote work\nRemote work requires manager approval and a home office audit.",
    "p-blank": "   ",
}


async def test_indexes_pages_without_kg_and_retrieves_the_right_page(tmp_path, monkeypatch):
    # The baseline must work on a machine without MinerU.
    from raganything.parser import MineruParser

    monkeypatch.setattr(MineruParser, "check_installation", lambda self: False)
    llm = NoKGLLM()
    pipeline = TextBaselinePipeline(
        working_dir=str(tmp_path / "store"), embedder=HashingEmbedder(), embedding_dim=DIM, llm=llm
    )
    await pipeline.aindex(list(PAGES), None, list(PAGES.values()))
    run = await pipeline.asearch(
        ["q1", "q2"], ["how many weeks of paid parental leave", "when are salary bands reviewed"]
    )
    await pipeline.aclose()

    assert pipeline.skipped_empty == ["p-blank"]
    assert max(run["q1"], key=run["q1"].get) == "p-leave"
    assert max(run["q2"], key=run["q2"].get) == "p-salary"
    assert set(run["q1"]) <= set(PAGES) - {"p-blank"}

    mean, _ = evaluate_run({"q1": {"p-leave": 1}, "q2": {"p-salary": 1}}, run)
    assert mean["recall_10"] == 1.0

    # No knowledge graph was built: the entity vector store stays empty.
    entities = json.loads((tmp_path / "store" / "vdb_entities.json").read_text(encoding="utf-8"))
    assert entities["data"] == []
