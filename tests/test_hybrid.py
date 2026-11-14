"""Hybrid pipeline logic with lightweight routes (BM25 stands in for the dense text route)."""

import json

import pytest

from helpers import PAGES, FakeLateEncoder, page_image
from mmrag.pipelines.hybrid import HybridPipeline
from mmrag.retrieval.bm25 import BM25Retriever
from mmrag.retrieval.fusion import ranked_ids
from mmrag.visual.retriever import VisualRetriever

IDS = list(PAGES)
TEXTS = list(PAGES.values())
IMAGES = [page_image(p) for p in IDS]
QIDS = ["leave", "chart", "tx"]
QUERIES = [
    "paid parental leave weeks",
    "headcount growth by region chart",
    "who books the TX-900 projector",
]


class ForcedAnalyzer:
    """Routes the chart page (3) and the table page (4) to the visual route."""

    def analyze(self, page_ids, texts, images=None, ocr_confidences=None):
        from mmrag.analysis.page_analyzer import PageAnalysis

        self.saw_images = images is not None
        return {
            p: PageAnalysis(p, "visual" if p in ("3", "4") else "text", ["forced"])
            for p in page_ids
        }


def build(routes=("text", "bm25", "visual"), **kwargs):
    made = {
        "text": BM25Retriever() if "text" in routes else None,
        "bm25": BM25Retriever() if "bm25" in routes else None,
        "visual": VisualRetriever(FakeLateEncoder()) if "visual" in routes else None,
    }
    kwargs.setdefault("analyzer", ForcedAnalyzer())
    return HybridPipeline(**made, **kwargs)


def test_pages_are_split_between_routes_and_bm25_covers_all():
    pipeline = build()
    pipeline.index(IDS, IMAGES, TEXTS)
    assert pipeline.routes["text"].ids == ["0", "1", "2", "5"]
    assert pipeline.routes["visual"].store.ids == ["3", "4"]
    assert pipeline.routes["bm25"].ids == IDS
    assert pipeline.analyzer.saw_images


def test_fusion_answers_questions_no_single_route_can():
    pipeline = build()
    pipeline.index(IDS, IMAGES, TEXTS)
    run = pipeline.search(QIDS, QUERIES)
    assert ranked_ids(run["leave"])[0] == "0"
    # Only the visual route can see the chart page. BM25 prefers the table page (it
    # contains "headcount"), so after fusion the chart page is in the top two.
    assert ranked_ids(pipeline.route_runs["visual"]["chart"])[0] == "3"
    assert "3" in ranked_ids(run["chart"])[:2]
    assert ranked_ids(run["tx"])[0] == "5"

    info = pipeline.info()
    assert set(info["route_runs"]) == {"text", "bm25", "visual", "fused"}
    assert "3" not in info["route_runs"]["text"]["chart"]
    assert info["page_routes"]["3"] == "visual" and info["page_routes"]["0"] == "text"
    assert pipeline.storage_bytes() == sum(info["storage_bytes_by_route"].values())


def test_ablation_removing_the_visual_route_loses_the_chart_page():
    pipeline = build(routes=("text", "bm25"))
    pipeline.index(IDS, IMAGES, TEXTS)
    run = pipeline.search(QIDS, QUERIES)
    assert "3" not in run["chart"]
    assert ranked_ids(run["leave"])[0] == "0"
    # Page 4 was routed visual, so with that route off only BM25 still reaches it.
    assert "4" not in pipeline.routes["text"].ids and "4" in pipeline.routes["bm25"].ids


def test_routing_all_gives_every_page_to_text_and_visual():
    pipeline = build(routing="all")
    pipeline.index(IDS, IMAGES, TEXTS)
    assert pipeline.routes["text"].ids == IDS
    assert pipeline.routes["visual"].store.ids == IDS
    assert set(pipeline.info()["page_routes"].values()) == {"both"}


def test_routed_index_is_smaller_than_indexing_everything_visually():
    routed, everything = build(), build(routing="all")
    routed.index(IDS, IMAGES, TEXTS)
    everything.index(IDS, IMAGES, TEXTS)
    assert routed.routes["visual"].storage_bytes() < everything.routes["visual"].storage_bytes()


def test_top_k_candidates_and_image_provider(tmp_path):
    def provider(ids):
        return ((pid, page_image(pid)) for pid in ids)

    pipeline = build(candidates=2, image_provider=provider, work_dir=tmp_path / "w")
    pipeline.index(IDS, None, TEXTS)
    run = pipeline.search(QIDS, QUERIES, top_k=3)
    assert all(len(r) <= 3 for r in run.values())
    assert all(len(r) <= 3 for r in pipeline.route_runs["bm25"].values())  # depth = max(2, 3)
    saved = json.loads((tmp_path / "w" / "page_analysis.json").read_text(encoding="utf-8"))
    assert saved["3"]["route"] == "visual"


def test_reranker_reorders_the_fused_candidates():
    class Reverse:
        def __init__(self):
            self.seen = []

        def rerank(self, query, candidates, top_n):
            self.seen.append((query, top_n, len(candidates)))
            order = ranked_ids(candidates)[:top_n]
            return {pid: float(i) for i, pid in enumerate(order)}

    reranker = Reverse()
    pipeline = build(reranker=reranker, rerank_top_n=3)
    pipeline.index(IDS, IMAGES, TEXTS)
    plain = build()
    plain.index(IDS, IMAGES, TEXTS)
    before = ranked_ids(plain.search(QIDS, QUERIES)["leave"])[:3]
    after = ranked_ids(pipeline.search(QIDS, QUERIES)["leave"])
    assert after == before[::-1]
    assert reranker.seen[0][:2] == ("paid parental leave weeks", 3)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="At least one route"):
        HybridPipeline()
    with pytest.raises(ValueError, match="routing"):
        build(routing="sometimes")
