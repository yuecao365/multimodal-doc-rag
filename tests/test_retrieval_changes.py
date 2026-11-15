"""Phase 6 switches: cross-encoder reranking, chunk headers, HyDE, Matryoshka."""

import json

import pytest

from helpers import FakeChat, FakeLateEncoder, HashingEmbedder, tokens
from mmrag.experiments.common import read_json
from mmrag.experiments.run import build_parser, run_experiment
from mmrag.rerank.rerankers import CrossEncoderReranker
from mmrag.retrieval.contextual import ContextualHeaders
from mmrag.retrieval.fusion import ranked_ids
from mmrag.retrieval.rewrite import HydeRewriter

HYBRID = ["--pipeline", "hybrid", "--min-text-chars", "20", "--no-analyzer-images"]


def overlap_scorer(query, texts):
    return [float(len(set(tokens(query)) & set(tokens(t)))) for t in texts]


@pytest.fixture
def run_dir(dataset_dir, tmp_path, monkeypatch):
    from mmrag.experiments import common

    monkeypatch.setattr(common, "ROOT", tmp_path)

    def make(name, *extra, **models):
        argv = ["--dataset", str(dataset_dir), "--date", "2000-01-01", "--name", name]
        argv += ["--work-dir", str(tmp_path / "work"), *HYBRID, *extra]
        models.setdefault("embedder", HashingEmbedder())
        models.setdefault("visual_encoder", FakeLateEncoder())
        return run_experiment(build_parser().parse_args(argv), **models)

    return make


def test_cross_encoder_reorders_text_pages_and_leaves_textless_pages_in_place():
    reranker = CrossEncoderReranker(scorer=overlap_scorer)
    reranker.prepare(
        ["a", "b", "c", "d"], ["salary bands", "", "parental leave weeks", "remote work"]
    )
    candidates = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    order = ranked_ids(reranker.rerank("weeks of parental leave", candidates, top_n=3))
    # "c" is the best text page; "b" has no text and keeps second place; "d" was not in the top 3.
    assert order == ["c", "b", "a", "d"]
    assert reranker.info()["pages_scored"] == 2


def test_cross_encoder_switch_runs_in_the_pipeline(run_dir):
    out = run_dir("ce", "--rerank", "cross-encoder", rerank_scorer=overlap_scorer)
    assert read_json(out / "cost.json")["after_search"]["rerank"]["queries"] == 4
    assert read_json(out / "metrics.json")["recall_10"] == 1.0


def test_contextual_headers_precede_every_chunk_and_are_written_once(tmp_path):
    def reply(system, parts):
        chunk = parts.split("<chunk>\n")[1]
        assert parts.index("<page number=") < parts.index("Chunk to situate")  # shared part first
        return "From the HR handbook: " + chunk.split()[0]

    ids = ["p1", "p2", "p3", "p4", "blank"]
    texts = ["alpha " * 30, "beta " * 30, "gamma " * 30, "delta " * 30, ""]
    kwargs = {
        "doc_ids": dict.fromkeys(ids, "handbook"),
        "page_numbers": {p: i for i, p in enumerate(ids)},
        "cache_path": tmp_path / "headers.json",
        "chunk_tokens": 20,
        "overlap_tokens": 5,
    }
    client = FakeChat(reply)
    headers = ContextualHeaders(client, **kwargs)
    out = headers(ids, texts)

    assert out[4] == ""  # a page without text is left alone
    assert out[0].startswith("From the HR handbook: alpha\n\nalpha")
    assert out[0].count("From the HR handbook") == headers.info()["chunks"] // 4 > 1
    # The first three pages form one group; the fourth page starts the next one.
    assert "gamma" in client.prompts[0][1] and "delta" not in client.prompts[0][1]

    again = ContextualHeaders(FakeChat([]), **kwargs)  # would fail if it called the model
    assert again(ids, texts) == out and again.info()["headers_written"] == 0


def test_contextual_headers_switch_indexes_the_new_text(run_dir, tmp_path):
    client = FakeChat(lambda system, parts: "About the zebra enclosure.")
    out = run_dir("ctx", "--contextual-headers", "--routes", "text,bm25", chat_client=client)
    cost = read_json(out / "cost.json")
    assert cost["after_index"]["text_transform"]["headers_written"] == client.calls == 5
    assert (tmp_path / "work" / "text_routed_ctx").is_dir()
    bm25 = json.loads((tmp_path / "work" / "bm25" / "index.json").read_text(encoding="utf-8"))
    assert "zebra" in json.dumps(bm25)


def test_hyde_searches_with_question_and_passage_and_caches_it(tmp_path):
    client = FakeChat(lambda system, parts: " Staff get sixteen weeks. ")
    rewriter = HydeRewriter(client, tmp_path / "hyde.json")
    texts = rewriter(["how long is leave", "how long is leave", "who books travel"])
    assert texts[0] == "how long is leave\nStaff get sixteen weeks." and client.calls == 2
    assert HydeRewriter(FakeChat([]), tmp_path / "hyde.json")(["who books travel"]) == texts[2:]


def test_hyde_switch_rewrites_for_routes_but_not_for_the_reranker(run_dir):
    seen = []

    def scorer(query, texts):
        seen.append(query)
        return overlap_scorer(query, texts)

    client = FakeChat(lambda system, parts: "hypothetical passage")
    out = run_dir(
        "hyde", "--hyde", "--rerank", "cross-encoder", chat_client=client, rerank_scorer=scorer
    )
    search_texts = read_json(out / "search_texts.json")
    assert search_texts["10"].endswith("\nhypothetical passage")
    # The visual route keeps the question: its ranking is the same as without HyDE.
    plain = run_dir("plain", "--rerank", "cross-encoder", rerank_scorer=overlap_scorer)
    visual = [read_json(d / "route_runs.json.gz")["visual"] for d in (out, plain)]
    assert visual[0] == visual[1]
    assert all("hypothetical" not in q for q in seen)
    assert read_json(out / "cost.json")["after_search"]["rewrite"]["passages_written"] == 4


def test_embedding_dim_builds_a_smaller_separate_text_index(run_dir, tmp_path):
    full = run_dir("full", "--routes", "text")
    small = run_dir("small", "--routes", "text", "--embedding-dim", "64")
    assert (tmp_path / "work" / "text_routed_d64").is_dir()
    size = lambda out: read_json(out / "cost.json")["index_storage_bytes"]
    assert size(small) < size(full)


def test_compression_table_compares_every_setting_with_the_uncompressed_route(
    dataset_dir, tmp_path
):
    from mmrag.data.vidore import load_vidore_subset
    from mmrag.experiments.chart import line_chart
    from mmrag.experiments.compression import compression_table, to_markdown
    from mmrag.visual.encode_cli import encode_dataset

    encode_dataset(FakeLateEncoder(), str(dataset_dir), tmp_path / "vectors", model="fake")
    data = load_vidore_subset(str(dataset_dir))
    rows = compression_table(
        data, str(tmp_path / "vectors"), tmp_path / "work", text_dims=(0, 64),
        embedder=HashingEmbedder(),
    )  # fmt: skip
    assert len(rows) == 2 + 9 + 9
    by = {(r["index"], r["setting"]): r for r in rows}
    assert (
        by["text", "full dims"]["difference"] == 0
        and by["visual", "float16, pool 1"]["p_value"] == 1
    )
    assert (
        by["visual", "binary, pool 3"]["kb_per_page"]
        < by["visual", "float16, pool 1"]["kb_per_page"]
    )
    assert by["text", "64 dims"]["kb_per_page"] < by["text", "full dims"]["kb_per_page"]
    # One chosen setting per index, never a reranked one.
    assert sorted(r["index"] for r in rows if r["chosen"]) == ["text", "visual"]
    assert len(to_markdown(rows).splitlines()) == len(rows) + 2

    series = {"visual": [(r["kb_per_page"], r["ndcg_cut_10"], r["setting"]) for r in rows[2:11]]}
    line_chart(series, tmp_path / "chart.svg", "t", "x", "y")
    assert (tmp_path / "chart.svg").read_text(encoding="utf-8").count("<circle") == 9
