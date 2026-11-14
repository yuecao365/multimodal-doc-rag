import json

from helpers import PAGES, VISIBLE, FakeChat, FakeLateEncoder, page_image
from mmrag.rerank.rerankers import LLMReranker, MaxSimReranker, merge_order
from mmrag.retrieval.fusion import ranked_ids
from mmrag.visual.store import MultiVectorStore

IDS = list(PAGES)
# First-stage order: 1, 2, 0, 5, 3, 4
CANDIDATES = {"1": 0.9, "2": 0.8, "0": 0.7, "5": 0.6, "3": 0.5, "4": 0.4}


def test_merge_order_puts_reranked_first_and_keeps_the_rest():
    merged = merge_order(["0", "1", "ghost"], CANDIDATES)
    assert ranked_ids(merged) == ["0", "1", "2", "5", "3", "4"]
    assert set(merged) == set(CANDIDATES)


def maxsim_reranker():
    encoder = FakeLateEncoder()
    store = MultiVectorStore.from_embeddings(
        IDS, [encoder._vectors(VISIBLE[p]) for p in IDS], "float16"
    )
    return MaxSimReranker(encoder, store)


def test_maxsim_promotes_the_matching_page_within_top_n():
    reranker = maxsim_reranker()
    out = reranker.rerank("paid parental leave weeks", CANDIDATES, top_n=3)
    order = ranked_ids(out)
    assert order[0] == "0"  # was third
    assert set(order[:3]) == {"0", "1", "2"}  # only the top 3 were rescored
    assert order[3:] == ["5", "3", "4"]  # the rest keep their order
    assert reranker.info()["pages_scored"] == 3
    assert reranker.storage_bytes() > 0


def test_maxsim_leaves_pages_without_vectors_in_place():
    reranker = maxsim_reranker()
    candidates = {"unknown-page": 1.0, **CANDIDATES}
    order = ranked_ids(reranker.rerank("salary bands reviewed", candidates, top_n=4))
    assert order[0] == "1"
    assert order.index("unknown-page") == 3  # after the 3 judged pages, before the rest


def llm_reranker(reply):
    client = FakeChat([reply])
    reranker = LLMReranker(client, excerpt_chars=30)
    reranker.prepare(IDS, list(PAGES.values()), lambda ids: ((p, page_image(p)) for p in ids))
    return reranker, client


def test_llm_reranker_applies_the_models_ranking():
    reply = json.dumps({"ranking": ["[PAGE 0]", "2", "not-a-page", "2"]})
    reranker, client = llm_reranker(reply)
    out = reranker.rerank("paid parental leave weeks", CANDIDATES, top_n=4)
    # Model order first (junk and repeats dropped), then the page it left out, then the rest.
    assert ranked_ids(out) == ["0", "2", "1", "5", "3", "4"]

    _, parts = client.prompts[0]
    texts = [p["text"] for p in parts if p["type"] == "text"]
    assert texts[0] == "[PAGE 1]" and texts[-1] == "Question: paid parental leave weeks"
    assert len(texts[1]) == 30  # excerpt is truncated
    assert reranker.info()["queries"] == 1


def test_llm_reranker_shows_textless_pages_as_images():
    reranker, client = llm_reranker(json.dumps({"ranking": ["3", "4"]}))
    out = reranker.rerank("headcount chart", {"4": 0.9, "3": 0.8}, top_n=2)
    assert ranked_ids(out) == ["3", "4"]
    _, parts = client.prompts[0]
    assert [p["type"] for p in parts] == ["text", "text", "text", "image_url", "text"]


def test_llm_reranker_keeps_first_stage_order_on_a_bad_reply():
    reranker, _ = llm_reranker("I would rank them as follows: first page 0...")
    out = reranker.rerank("q", CANDIDATES, top_n=4)
    assert ranked_ids(out) == ranked_ids(CANDIDATES)
    assert reranker.info()["unparsed_replies"] == 1


def test_llm_reranker_skips_the_call_for_a_single_candidate():
    reranker, client = llm_reranker("unused")
    assert ranked_ids(reranker.rerank("q", {"1": 0.5}, top_n=5)) == ["1"]
    assert client.calls == 0
