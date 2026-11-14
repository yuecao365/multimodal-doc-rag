import pytest

from mmrag.eval.provenance import citation_page_metrics, iou, mean_of, region_iou
from mmrag.generation.answer import Answer, Citation
from mmrag.generation.citations import (
    NO_PAGE_TEXT,
    NO_QUOTE,
    PAGE_NOT_IN_CONTEXT,
    QUOTE_NOT_FOUND,
    VERIFIED,
    normalize,
    quote_match,
    summarize_reports,
    verify_citations,
)

PAGE = (
    "## Parental leave\n\n"
    "Employees receive **sixteen weeks** of paid parental leave,\n"
    "which may be taken within “the first year” after birth.\n\n"
    "| Region | Weeks |\n|---|---|\n| EMEA | 16 |\n| APAC | 14 |\n"
)
TEXTS = {"7": PAGE, "8": "Salary bands are reviewed every April.", "9": ""}


def test_normalize_strips_markup_and_typography():
    assert normalize("**Sixteen   weeks** — “paid”") == 'sixteen weeks - "paid"'


def test_quote_match_tolerates_formatting_but_not_invention():
    assert quote_match("sixteen weeks of paid parental leave", PAGE) == 1.0
    # Line break and bold markers in the source, straight quotes in the quote.
    assert quote_match('leave, which may be taken within "the first year"', PAGE) == 1.0
    # A small slip (plural) still counts as found...
    assert quote_match("Employees receive sixteen week of paid parental leave", PAGE) > 0.9
    # ...a fabricated sentence does not.
    assert quote_match("Employees receive twenty-six weeks of unpaid sabbatical", PAGE) < 0.85
    assert quote_match("completely unrelated statement about pensions", PAGE) < 0.6
    assert quote_match("", PAGE) == 0.0 and quote_match("x", "") == 0.0


def test_each_citation_gets_the_right_status():
    answer = Answer(
        answer="Sixteen weeks.",
        citations=[
            Citation("7", "sixteen weeks of paid parental leave"),
            Citation("7", "employees receive forty weeks of leave in all regions worldwide"),
            Citation("8", ""),
            Citation("9", "some words read off the chart"),
            Citation("42", "sixteen weeks"),
        ],
    )
    report = verify_citations(answer, ["7", "8", "9"], TEXTS)
    assert [c.status for c in report.checks] == [
        VERIFIED,
        QUOTE_NOT_FOUND,
        NO_QUOTE,
        NO_PAGE_TEXT,
        PAGE_NOT_IN_CONTEXT,
    ]
    assert report.valid_pages == ["7", "8", "9"]  # 42 was never shown
    assert not report.all_ok
    summary = report.to_dict()
    assert summary["n_citations"] == 5 and summary[PAGE_NOT_IN_CONTEXT] == 1


def test_a_page_that_is_relevant_but_was_not_shown_is_still_rejected():
    answer = Answer("x", [Citation("8", "Salary bands are reviewed every April")])
    report = verify_citations(answer, ["7"], TEXTS)
    assert report.checks[0].status == PAGE_NOT_IN_CONTEXT


def test_all_ok_requires_at_least_one_citation():
    assert not verify_citations(Answer("no sources"), ["7"], TEXTS).all_ok
    good = verify_citations(Answer("x", [Citation("7", "paid parental leave")]), ["7"], TEXTS)
    assert good.all_ok


def test_summary_over_answers():
    reports = [
        verify_citations(Answer("x", [Citation("7", "paid parental leave")]), ["7"], TEXTS),
        verify_citations(Answer("y", [Citation("42", "q"), Citation("8", "")]), ["8"], TEXTS),
        verify_citations(Answer("z"), ["7"], TEXTS),
    ]
    s = summarize_reports(reports)
    assert s["n_answers"] == 3 and s["n_citations"] == 3
    assert s["answers_with_citations"] == pytest.approx(2 / 3)
    assert s["answers_all_citations_ok"] == pytest.approx(1 / 3)
    assert s["citation_verified_rate"] == pytest.approx(1 / 3)
    assert s["citation_page_not_in_context_rate"] == pytest.approx(1 / 3)


def test_citation_page_metrics():
    qrels = {"a": 1, "b": 2, "c": 0}
    m = citation_page_metrics(["a", "x", "a"], qrels)
    assert m["cited_pages"] == 2.0
    assert m["citation_precision"] == 0.5 and m["citation_recall"] == 0.5
    assert m["cites_a_relevant_page"] == 1.0
    none = citation_page_metrics([], qrels)
    assert none["citation_precision"] == 0.0 and none["citation_f1"] == 0.0
    assert citation_page_metrics(["c"], qrels)["cites_a_relevant_page"] == 0.0  # score 0
    assert mean_of([m, none])["citation_precision"] == 0.25


def test_region_iou_converts_normalised_boxes_to_pixels():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    gold = [{"x1": 100, "y1": 200, "x2": 300, "y2": 400, "annotator": 0}]
    # The same box expressed in 0..1000 coordinates of a 1000x2000 page.
    assert region_iou([100, 100, 300, 200], gold, (1000, 2000)) == pytest.approx(1.0)
    assert region_iou([300, 200, 100, 100], gold, (1000, 2000)) == pytest.approx(
        1.0
    )  # corners swapped
    assert region_iou([600, 600, 900, 900], gold, (1000, 2000)) == 0.0
    assert region_iou([0, 0, 10, 10], [], (1000, 2000)) == 0.0
