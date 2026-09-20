"""Unit tests for the text-to-handwriting renderer (src/renderer/text_renderer.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.generator.glyph import GlyphBitmap, GlyphMetrics, make_bitmap
from src.renderer.text_renderer import TextRenderer

RENDERER_CFG = {
    "renderer": {
        "target_x_height_px": 40,
        "char_spacing_px": 4,
        "word_spacing_px": 20,
        "line_height_multiplier": 1.6,
        "baseline_jitter_px": 0,
        "rotation_jitter_deg": 0.0,
        "canvas_margin_px": 10,
    }
}


def make_alphabet() -> dict[str, list[GlyphBitmap]]:
    """26 identical block glyphs, 25 px tall / 15 wide, standing on the baseline, x-height 25."""
    alphabet = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        img = np.full((25, 15), 255, dtype=np.uint8)
        alphabet[c] = [make_bitmap(img, x_height=25.0, baseline_row=25.0, spacing=5.0),
                       make_bitmap(img.copy(), x_height=25.0, baseline_row=25.0, spacing=5.0)]
    return alphabet


def test_render_line_produces_nonempty_image():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    img = renderer.render_line("hello")
    assert img.ndim == 2
    assert img.sum() > 0


def test_render_line_width_grows_with_text_length():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    short_img = renderer.render_line("hi")
    long_img = renderer.render_line("hello world")
    assert long_img.shape[1] > short_img.shape[1]


def test_render_text_wraps_long_lines():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    long_text = " ".join(["word"] * 60)
    img = renderer.render_text(long_text, max_line_width_px=400)
    # A wrapped multi-line render should be taller than a single line.
    single_line = renderer.render_line("word")
    assert img.shape[0] > single_line.shape[0]


def test_every_line_has_the_same_height_so_wrapped_lines_share_a_baseline():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    assert renderer.render_line("a").shape[0] == renderer.render_line("hello world").shape[0]


def test_missing_glyph_does_not_crash():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    img = renderer.render_line("hello! 123")  # punctuation/digits unsupported
    assert img.ndim == 2


def test_render_to_bgr_on_white_has_white_background_corners():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    img = renderer.render_to_bgr_on_white("hi")
    assert img.shape[2] == 3
    assert tuple(img[0, 0]) == (255, 255, 255)


def test_empty_alphabet_is_rejected():
    with pytest.raises(ValueError):
        TextRenderer({}, RENDERER_CFG)


def test_uppercase_falls_back_to_scaled_lowercase_glyph_on_the_same_baseline():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    lower, upper = renderer.render_line("a"), renderer.render_line("A")
    rows = lambda im: np.nonzero(im.max(axis=1))[0]  # noqa: E731
    assert rows(upper)[-1] == pytest.approx(rows(lower)[-1], abs=1)      # same baseline (bottom row)
    assert rows(upper)[0] < rows(lower)[0]                               # but taller
