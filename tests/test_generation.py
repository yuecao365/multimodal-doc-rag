from helpers import FakeChat, answer_json, page_image
from mmrag.generation.answer import generate_answer, parse_answer
from mmrag.generation.context import assemble_context, estimate_tokens

TEXTS = {"a": "alpha " * 100, "b": "beta " * 100, "c": "", "d": "delta " * 100}


def test_estimate_tokens_counts_cjk_per_character():
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("多模态检索") == 5


def test_takes_pages_in_rank_order_up_to_max_pages():
    ctx = assemble_context(["b", "a", "d"], TEXTS, max_pages=2)
    assert [c.page_id for c in ctx] == ["b", "a"]
    assert all(c.form == "text" for c in ctx)


def test_form_follows_route_and_falls_back_when_unavailable():
    images = {"a": page_image("1"), "c": page_image("2")}
    ctx = assemble_context(
        ["a", "b", "c"], TEXTS, images=images, forms={"a": "image", "b": "image"}
    )
    forms = {c.page_id: c.form for c in ctx}
    # a: image as asked; b: asked for image but has none; c: no text, so image.
    assert forms == {"a": "image", "b": "text", "c": "image"}


def test_token_budget_truncates_text_and_skips_what_does_not_fit():
    per_page = estimate_tokens(TEXTS["a"].strip())
    budget = per_page + 100
    ctx = assemble_context(["a", "b", "d"], TEXTS, token_budget=budget, min_text_tokens=50)
    # a fits whole, b is cut to the 100 tokens left, d no longer fits at all.
    assert [c.page_id for c in ctx] == ["a", "b"]
    assert not ctx[0].truncated and ctx[1].truncated
    assert sum(c.tokens for c in ctx) <= budget

    strict = assemble_context(["a", "b", "d"], TEXTS, token_budget=budget, min_text_tokens=150)
    assert [c.page_id for c in strict] == ["a"]  # too little room left to be worth a page


def test_image_over_budget_is_sent_as_text():
    images = {"a": page_image("1")}
    ctx = assemble_context(
        ["a"], TEXTS, images=images, forms={"a": "image"}, token_budget=500, image_tokens=1500
    )
    assert ctx[0].form == "text"


def test_generate_builds_prompt_with_page_ids_and_parses_citations():
    ctx = assemble_context(
        ["a", "c"], TEXTS, images={"c": page_image("2")}, labels={"a": "handbook p.3"}
    )
    client = FakeChat([answer_json("Sixteen weeks.", [("a", "alpha alpha"), ("[PAGE c]", "")])])
    answer = generate_answer(client, "how long?", ctx)

    assert answer.answer == "Sixteen weeks."
    assert answer.cited_pages == ["a", "c"]
    _, parts = client.prompts[0]
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "text", "text", "image_url", "text"]
    assert parts[0]["text"] == "[PAGE a] (handbook p.3)"
    assert parts[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[-1]["text"] == "Question: how long?"


def test_parse_handles_code_fences_regions_and_garbage():
    fenced = '```json\n{"answer": "x", "citations": [{"page": 7, "region": [0, 0, 500, 500]}]}\n```'
    answer = parse_answer(fenced)
    assert answer.parsed and answer.citations[0].page == "7"
    assert answer.citations[0].region == [0, 0, 500, 500]

    plain = parse_answer("I think it is sixteen weeks.")
    assert not plain.parsed and plain.answer == "I think it is sixteen weeks."
    assert plain.citations == []


def test_no_context_makes_no_model_call():
    client = FakeChat([])
    answer = generate_answer(client, "q", [])
    assert client.calls == 0 and answer.citations == []
