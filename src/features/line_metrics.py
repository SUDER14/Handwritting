"""Writer x-height and baseline, estimated from labelled glyph boxes.

The segmenter's `baseline_y` / `x_height_top_y` are medians of ALL glyph bottoms/tops, which an
ascender ('k') or descender ('q') drags around. With the character labels we can do better:
letters that live entirely in the x-height band (a c e m n o r s u v w x z) give the x-height as
their box height and the baseline as their box bottom, directly. Only when a sample has none of
them (e.g. "fly") do we fall back to per-letter typographic priors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.generator.glyph import GlyphPriors, glyph_class


@dataclass(frozen=True)
class LineMetrics:
    baseline_y: float    # row of the writing baseline in the source line image (y down)
    x_height: float      # px
    source: str          # "x_height_letters" (measured) | "priors" (estimated from letter-shape priors)
    n_used: int          # glyphs the estimate is based on


def estimate_line_metrics(chars, glyphs, priors: GlyphPriors | None = None) -> LineMetrics:
    """`glyphs` need a `.bbox = (x, y, w, h)`; `chars[i]` labels `glyphs[i]`."""
    priors = priors or GlyphPriors()
    pairs = [(c.lower(), g.bbox) for c, g in zip(chars, glyphs)]
    if not pairs:
        raise ValueError("cannot estimate line metrics from zero glyphs")

    measured = [b for c, b in pairs if glyph_class(c) == "x"]
    if measured:
        return LineMetrics(
            baseline_y=float(np.median([y + h for _, y, _, h in measured])),
            x_height=float(np.median([h for _, _, _, h in measured])),
            source="x_height_letters", n_used=len(measured),
        )

    x_heights, baselines = [], []
    for c, (_, y, _, h) in pairs:
        xh = h / priors.total_height_ratio(c)
        x_heights.append(xh)
        baselines.append((y + h) - priors.descender_depth(c) * xh)
    return LineMetrics(
        baseline_y=float(np.median(baselines)), x_height=float(np.median(x_heights)),
        source="priors", n_used=len(pairs),
    )
