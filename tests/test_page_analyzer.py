import numpy as np
from PIL import Image

from mmrag.analysis.page_analyzer import (
    AnalyzerConfig,
    PageAnalyzer,
    image_features,
    route_counts,
    text_features,
)

PROSE = "The committee reviewed the annual workforce report in detail. " * 12
TABLE = "\n".join(
    ["| Year | Headcount | Attrition |", "|---|---|---|"]
    + [f"| {2000 + i} | {400 + i * 13} | {8 + i % 5}.{i}% |" for i in range(30)]
)


def white_page(size=(400, 520)):
    return Image.fromarray(np.full((size[1], size[0], 3), 255, dtype=np.uint8))


def text_page():
    arr = np.asarray(white_page()).copy()
    arr[40:480:12, 40:360] = 30  # thin dark lines of "text"
    return Image.fromarray(arr)


def chart_page():
    arr = np.asarray(white_page()).copy()
    arr[100:400, 60:160] = (200, 40, 40)
    arr[180:400, 180:280] = (40, 90, 200)
    return Image.fromarray(arr)


def test_text_features_separate_prose_from_tables():
    prose = text_features(PROSE)
    assert prose["table_ratio"] == 0.0 and prose["text_chars"] > 500

    table = text_features(TABLE)
    assert table["table_ratio"] == 1.0

    mixed = text_features(PROSE + "\n\n" + TABLE)
    assert 0.0 < mixed["table_ratio"] < 1.0
    assert text_features("")["text_chars"] == 0.0


def test_figure_markers_are_counted():
    text = "Figure 3: Share of remote workers\n\n![chart](fig3.png)\n\nSome commentary."
    assert text_features(text)["n_figures"] == 2.0
    assert text_features("We figure this out later.")["n_figures"] == 0.0


def test_image_features_tell_coloured_graphics_from_text():
    assert image_features(white_page()) == {"color_ratio": 0.0, "ink_ratio": 0.0}
    text = image_features(text_page())
    chart = image_features(chart_page())
    assert text["color_ratio"] == 0.0 and text["ink_ratio"] > 0.0
    assert chart["color_ratio"] > 0.2


def test_routing_rules_and_reasons():
    analyzer = PageAnalyzer()
    assert analyzer.analyze_page("p", PROSE, text_page()).route == "text"
    assert analyzer.analyze_page("p", PROSE).reasons == []

    cases = {
        "little_text": analyzer.analyze_page("p", "Annual report 2024"),
        "table_heavy": analyzer.analyze_page("p", TABLE),
        "coloured_graphics": analyzer.analyze_page("p", PROSE, chart_page()),
        "figure": analyzer.analyze_page("p", "Figure 2: Attrition by region\n" + PROSE[:400]),
        "low_ocr_confidence": analyzer.analyze_page("p", PROSE, ocr_confidence=0.55),
    }
    for reason, analysis in cases.items():
        assert analysis.route == "visual", reason
        assert reason in analysis.reasons, reason


def test_ink_without_text_marks_scans():
    # A page mostly covered in dark ink whose parser found only a little text.
    dark = Image.fromarray(np.full((520, 400, 3), 90, dtype=np.uint8))
    analysis = PageAnalyzer().analyze_page("p", PROSE[:400], dark)
    assert "ink_without_text" in analysis.reasons


def test_thresholds_are_configurable():
    strict = PageAnalyzer(AnalyzerConfig(min_text_chars=5000))
    assert strict.analyze_page("p", PROSE).route == "visual"
    lenient = PageAnalyzer(AnalyzerConfig(max_table_ratio=1.0))
    assert lenient.analyze_page("p", TABLE).route == "text"


def test_analyze_corpus_with_partial_images_and_summary():
    analyzer = PageAnalyzer()
    ids = ["a", "b", "c"]
    texts = [PROSE, PROSE, ""]
    analyses = analyzer.analyze(ids, texts, images=[("b", chart_page()), ("zzz", chart_page())])
    assert list(analyses) == ids
    assert [analyses[i].route for i in ids] == ["text", "visual", "visual"]
    assert "color_ratio" not in analyses["a"].features  # judged on text alone
    summary = route_counts(analyses)
    assert summary["n_text"] == 1 and summary["n_visual"] == 2
    assert summary["visual_reasons"] == {
        "coloured_graphics": 1,
        "ink_without_text": 1,
        "little_text": 1,
    }
