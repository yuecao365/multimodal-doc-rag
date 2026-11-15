"""The retrieval agent on the tiny dataset, driven by a scripted model."""

import json
from types import SimpleNamespace

import pytest

from helpers import FakeChat, FakeLateEncoder, HashingEmbedder, answer_json
from mmrag.experiments.answer import answer_run
from mmrag.experiments.common import read_json, read_jsonl
from mmrag.experiments.run import build_parser, run_experiment
from mmrag.pipelines.agent import snippet
from mmrag.retrieval.fusion import ranked_ids

HYBRID = ["--pipeline", "hybrid", "--min-text-chars", "20", "--no-analyzer-images", "--agent"]


class ScriptedAgent:
    """Chat model stand-in: ``script(question, turn, messages)`` -> [(tool, arguments)]."""

    def __init__(self, script):
        self.script = script
        self.requests: list[dict] = []

    def chat(self, messages, tools=None, tool_choice=None):
        question = messages[1]["content"].split("\n")[0].removeprefix("Question: ")
        turn = sum(1 for m in messages if m["role"] == "assistant")
        self.requests.append({"question": question, "choice": tool_choice, "messages": messages})
        if isinstance(tool_choice, dict):  # the pipeline forces the end
            calls = [("finish", {"page_ids": ["1"]})]
        else:
            calls = self.script(question, turn, messages)
        tool_calls = [
            SimpleNamespace(
                id=f"call_{turn}_{i}",
                function=SimpleNamespace(name=name, arguments=json.dumps(args)),
            )
            for i, (name, args) in enumerate(calls)
        ]
        return SimpleNamespace(content=None, tool_calls=tool_calls), 100

    def usage(self):
        return {"model": "scripted", "calls": len(self.requests)}


@pytest.fixture
def run_agent(dataset_dir, tmp_path, monkeypatch):
    from mmrag.experiments import common

    monkeypatch.setattr(common, "ROOT", tmp_path)

    def make(script, *extra):
        argv = ["--dataset", str(dataset_dir), "--date", "2000-01-01", "--name", "agent"]
        argv += ["--work-dir", str(tmp_path / "work"), *HYBRID, *extra]
        client = ScriptedAgent(script)
        out = run_experiment(
            build_parser().parse_args(argv),
            embedder=HashingEmbedder(),
            visual_encoder=FakeLateEncoder(),
            chat_client=client,
        )
        return out, client

    return make


def test_agent_searches_again_and_its_pages_head_the_ranking(run_agent):
    def script(question, turn, messages):
        if "TX-900" in question and turn == 0:
            return [("keyword_search", {"query": "TX-900 projector"})]
        if "TX-900" in question:
            found = json.loads(messages[-1]["content"])
            return [("finish", {"page_ids": [found[0]["page_id"], "2", "no-such-page"]})]
        return [("finish", {"page_ids": []})]  # the first results are enough: keep them

    out, client = run_agent(script)
    run, selected = read_json(out / "run.json"), read_json(out / "selected_pages.json")
    assert selected["12"] == ["5", "2"]  # unknown ids are dropped
    assert ranked_ids(run["12"])[:2] == ["5", "2"] and len(run["12"]) > 2
    assert selected["10"] == [] and ranked_ids(run["10"])[0] == "0"  # one-shot order kept

    steps = read_json(out / "agent_steps.json")["12"]
    assert [s["tool"] for s in steps["steps"]] == ["keyword_search", "finish"]
    assert steps["tool_calls"] == 1 and not steps["forced_finish"]
    agent = read_json(out / "cost.json")["after_search"]["agent"]
    assert agent["questions"] == 4 and agent["tool_calls_per_question"] == 0.25
    # The first message shows the one-shot results with snippets.
    first = client.requests[0]["messages"][1]["content"]
    assert "Results of the first search" in first and '"snippet"' in first

    # Answers are generated from the selected pages only.
    generator = FakeChat(lambda system, parts: answer_json("x", []))
    answer_run(out, generator, judge=None)
    records = {r["query_id"]: r for r in read_jsonl(out / "answers.jsonl")}
    assert [c["page"] for c in records["12"]["context"]] == ["5", "2"]
    assert len(records["10"]["context"]) > 2


def test_agent_is_stopped_at_the_tool_call_limit(run_agent):
    def script(question, turn, messages):
        return [("text_search", {"query": f"attempt {turn}"})]

    out, client = run_agent(script, "--agent-max-tool-calls", "2")
    steps = read_json(out / "agent_steps.json")
    assert all(s["tool_calls"] == 2 and s["forced_finish"] for s in steps.values())
    assert read_json(out / "selected_pages.json")["10"] == ["1"]
    forced = [r for r in client.requests if isinstance(r["choice"], dict)]
    assert len(forced) == 4 and forced[0]["choice"]["function"]["name"] == "finish"


def test_agent_reads_a_visual_page_as_an_image(run_agent):
    def script(question, turn, messages):
        if turn == 0:
            return [("read_page", {"page_id": "3"}), ("read_page", {"page_id": "0"})]
        return [("finish", {"page_ids": ["3"]})]

    _, client = run_agent(script)
    after = client.requests[1]["messages"]
    tool_replies = [m["content"] for m in after if m["role"] == "tool"]
    assert tool_replies[0] == "The page image follows." and "parental leave" in tool_replies[1]
    image_message = [m for m in after if m["role"] == "user"][-1]
    assert image_message["role"] == "user" and image_message["content"][1]["type"] == "image_url"


def test_visual_search_is_offered_only_when_new_queries_can_be_encoded():
    from mmrag.pipelines.agent import AgentSearch
    from mmrag.visual.encoder import PrecomputedQueryEncoder

    def tool_names(encoder):
        visual = SimpleNamespace(encoder=encoder)
        pipeline = SimpleNamespace(routes={"bm25": SimpleNamespace(), "visual": visual})
        agent = AgentSearch(pipeline, client=None)
        return [t["function"]["name"] for t in agent.tools()]

    stored_only = PrecomputedQueryEncoder(store=None)
    assert tool_names(stored_only) == ["keyword_search", "read_page", "finish"]
    with_worker = PrecomputedQueryEncoder(store=None, fallback=object())
    assert "visual_search" in tool_names(with_worker)


def test_snippet_shows_the_part_of_the_page_with_the_query_words():
    text = "intro " * 80 + "the projector model TX-900 must be booked " + "outro " * 80
    assert "projector" in snippet(text, "who books the projector", width=80)
    assert snippet("short page", "anything") == "short page"
