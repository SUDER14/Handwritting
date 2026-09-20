"""Unit tests for the text-to-handwriting renderer (src/renderer/text_renderer.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.renderer.text_renderer import TextRenderer

RENDERER_CFG = {
    "renderer": {
        "char_spacing_px": 4,
        "word_spacing_px": 20,
        "line_height_multiplier": 1.6,
        "baseline_jitter_px": 1,
        "rotation_jitter_deg": 1.0,
        "canvas_margin_px": 10,
    }
}


def make_alphabet() -> dict[str, list[np.ndarray]]:
    alphabet = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        img = np.zeros((40, 25), dtype=np.uint8)
        img[5:35, 5:20] = 255
        alphabet[c] = [img, img.copy()]
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


def test_missing_glyph_does_not_crash():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    img = renderer.render_line("hello! 123")  # punctuation/digits unsupported
    assert img.ndim == 2


def test_render_to_bgr_on_white_has_white_background_corners():
    renderer = TextRenderer(make_alphabet(), RENDERER_CFG)
    img = renderer.render_to_bgr_on_white("hi")
    assert img.shape[2] == 3
    assert tuple(img[0, 0]) == (255, 255, 255)
