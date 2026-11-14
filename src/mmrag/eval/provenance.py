"""Provenance metrics: are the cited pages (and optionally regions) the right ones?"""

from __future__ import annotations


def citation_page_metrics(cited_pages: list[str], qrels: dict[str, int]) -> dict[str, float]:
    """Compare cited pages with the pages judged relevant for the query."""
    relevant = {p for p, score in qrels.items() if score > 0}
    cited = list(dict.fromkeys(cited_pages))
    hits = [p for p in cited if p in relevant]
    precision = len(hits) / len(cited) if cited else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cited_pages": float(len(cited)),
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "cites_a_relevant_page": float(bool(hits)),
    }


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = width * height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def region_iou(
    predicted: list[float],
    gold_boxes: list[dict],
    image_size: tuple[int, int],
    scale: float = 1000.0,
) -> float:
    """Best IoU between a predicted region and the annotated evidence boxes.

    ``predicted`` is [x1, y1, x2, y2] in 0..``scale`` coordinates relative to the
    page; ``gold_boxes`` are the dataset's pixel boxes ({x1, y1, x2, y2}) on an
    image of ``image_size`` (width, height).
    """
    width, height = image_size
    x1, y1, x2, y2 = predicted
    box = (
        min(x1, x2) / scale * width,
        min(y1, y2) / scale * height,
        max(x1, x2) / scale * width,
        max(y1, y2) / scale * height,
    )
    return max(
        (iou(box, (g["x1"], g["y1"], g["x2"], g["y2"])) for g in gold_boxes),
        default=0.0,
    )


def mean_of(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
