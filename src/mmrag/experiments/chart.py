"""A small SVG line chart, written by hand so that no plotting library is needed."""

from __future__ import annotations

import math
from pathlib import Path

COLORS = ("#2563eb", "#d97706", "#059669", "#7c3aed")
WIDTH, HEIGHT = 760, 440
LEFT, RIGHT, TOP, BOTTOM = 70, 170, 40, 60


def _ticks(low: float, high: float, n: int = 5) -> list[float]:
    step = (high - low) / n
    return [low + i * step for i in range(n + 1)]


def line_chart(
    series: dict[str, list[tuple[float, float, str]]],
    path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Write one line per series. Points are (x, y, label); x is on a log scale."""
    points = [p for line in series.values() for p in line]
    x_low = math.log10(min(p[0] for p in points)) - 0.1
    x_high = math.log10(max(p[0] for p in points)) + 0.1
    y_low = math.floor(min(p[1] for p in points) * 20) / 20
    y_high = max(math.ceil(max(p[1] for p in points) * 20) / 20, y_low + 0.05)

    def sx(x: float) -> float:
        return LEFT + (math.log10(x) - x_low) / (x_high - x_low) * (WIDTH - LEFT - RIGHT)

    def sy(y: float) -> float:
        return HEIGHT - BOTTOM - (y - y_low) / (y_high - y_low) * (HEIGHT - TOP - BOTTOM)

    text = 'font-family="sans-serif" font-size="12" fill="#374151"'
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
        f'<text x="{LEFT}" y="22" {text} font-weight="bold">{title}</text>',
    ]
    for y in _ticks(y_low, y_high):
        out.append(
            f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" '
            'stroke="#e5e7eb"/>'
            f'<text x="{LEFT - 8}" y="{sy(y) + 4:.1f}" text-anchor="end" {text}>{y:.2f}</text>'
        )
    decade = math.floor(x_low)
    while decade <= x_high:
        for mult in (1, 2, 5):
            x = mult * 10**decade
            if x_low <= math.log10(x) <= x_high:
                out.append(
                    f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{TOP}" y2="{HEIGHT - BOTTOM}" '
                    'stroke="#f3f4f6"/>'
                    f'<text x="{sx(x):.1f}" y="{HEIGHT - BOTTOM + 18}" text-anchor="middle" {text}>'
                    f"{x:g}</text>"
                )
        decade += 1
    out.append(
        f'<text x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 14}" text-anchor="middle" {text}>'
        f"{x_label}</text>"
        f'<text transform="translate(18 {(TOP + HEIGHT - BOTTOM) / 2}) rotate(-90)" '
        f'text-anchor="middle" {text}>{y_label}</text>'
    )
    for i, (name, line) in enumerate(series.items()):
        color = COLORS[i % len(COLORS)]
        line = sorted(line)
        path_d = " ".join(
            f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y, _) in enumerate(line)
        )
        out.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, label in line:
            out.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>'
                f'<text x="{sx(x) + 5:.1f}" y="{sy(y) - 6:.1f}" font-family="sans-serif" '
                f'font-size="9" fill="{color}">{label}</text>'
            )
        legend_y = TOP + 10 + i * 20
        out.append(
            f'<rect x="{WIDTH - RIGHT + 16}" y="{legend_y - 9}" width="12" height="12" '
            f'fill="{color}"/>'
            f'<text x="{WIDTH - RIGHT + 34}" y="{legend_y + 1}" {text}>{name}</text>'
        )
    out.append("</svg>")
    Path(path).write_text("\n".join(out), encoding="utf-8")
