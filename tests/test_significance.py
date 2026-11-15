"""Paired tests, breakdowns and the spend log."""

from mmrag.data.vidore import load_vidore_subset
from mmrag.eval.breakdown import breakdown, relevant_bin
from mmrag.eval.significance import mean_ci, paired_test
from mmrag.experiments import spend


def test_paired_test_separates_a_real_gain_from_noise():
    old = {str(i): 0.5 for i in range(60)}
    better = {q: v + (0.2 if int(q) % 3 else 0.0) for q, v in old.items()}
    noise = {q: v + (0.2 if int(q) % 2 else -0.2) for q, v in old.items()}

    gain = paired_test(better, old)
    assert abs(gain["mean"] - 0.1333) < 1e-3 and gain["p_value"] < 0.001
    assert gain["ci95"][0] > 0
    flat = paired_test(noise, old)
    assert abs(flat["mean"]) < 1e-9 and flat["p_value"] > 0.5
    assert flat["ci95"][0] < 0 < flat["ci95"][1]


def test_mean_ci_brackets_the_mean():
    report = mean_ci([0.0, 1.0] * 50)
    assert report["mean"] == 0.5 and report["ci95"][0] < 0.5 < report["ci95"][1]


def test_breakdown_groups_by_type_and_relevant_pages(dataset_dir):
    data = load_vidore_subset(str(dataset_dir), language="english")
    data.query_types = {"10": ["numerical"], "11": ["numerical", "extractive"]}
    report = breakdown({"10": 1.0, "11": 0.0, "12": 1.0}, data)
    assert report["query_type"]["numerical"] == {"n": 2, "mean": 0.5}
    assert report["query_type"]["extractive"] == {"n": 1, "mean": 0.0}
    assert report["relevant_pages"]["1"]["n"] == 3
    assert [relevant_bin(n) for n in (1, 4, 5, 9, 10, 40)] == [
        "1",
        "2-4",
        "5-9",
        "5-9",
        "10+",
        "10+",
    ]


def test_spend_log_sums_nested_costs_and_enforces_the_cap(tmp_path, monkeypatch):
    from mmrag.experiments import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    usage = {"generator": {"cost_usd": 0.25}, "rerank": {"usage": {"cost_usd": 0.5}}, "judge": None}
    assert spend.record("run_a", "answers", usage) == 0.75
    assert spend.record("run_b", "nothing paid", {"calls": 3}) == 0.0
    assert spend.spent() == 0.75
    assert "| run_a | answers | 0.7500 | 0.7500 |" in (tmp_path / "results/spend.md").read_text()

    spend.check_budget()
    monkeypatch.setenv("BUDGET_USD", "0.5")
    try:
        spend.check_budget()
    except RuntimeError as error:
        assert "cap reached" in str(error)
    else:
        raise AssertionError("the cap was not enforced")
