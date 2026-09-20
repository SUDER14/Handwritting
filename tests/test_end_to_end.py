"""End-to-end test (section 19): handwriting image -> preprocessing ->
segmentation -> style extraction -> alphabet generation -> text generation
-> output image.

Uses the baseline (non-neural) generator so this test has no dependency on
a trained checkpoint existing, and stays fast enough to run in CI/on every
commit. The neural backend is covered separately by
tests/test_style_encoder_model.py (architecture) and manual evaluation via
scripts/generate_alphabet.py once a checkpoint is trained.
"""
from __future__ import annotations

import string

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.pipeline import analyze_sample, generate_alphabet_baseline
from src.renderer.text_renderer import TextRenderer
from src.utils.config import load_config


def _synthetic_word_image(word: str) -> np.ndarray:
    font = None
    for path in ["C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/comic.ttf"]:
        try:
            font = ImageFont.truetype(path, 80)
            break
        except OSError:
            continue
    if font is None:
        pytest.skip("No suitable system font available to synthesize a test sample.")

    img = Image.new("L", (700, 220), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((700 - w) / 2 - bbox[0], (220 - h) / 2 - bbox[1]), word, fill=0, font=font)
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_end_to_end_baseline_pipeline(config):
    word = "quick"
    image_bgr = _synthetic_word_image(word)

    # Stage 1: preprocessing + segmentation + recognition + feature extraction.
    analysis = analyze_sample(image_bgr, word, config)
    assert analysis.chars == list(word)
    assert len(analysis.segmentation.glyphs) == len(word)
    assert analysis.style_profile.mean_height_px > 0

    # Stage 2: full alphabet generation (hybrid: observed + baseline-synthesized).
    alphabet = generate_alphabet_baseline(analysis, config)
    assert set(alphabet.keys()) == set(string.ascii_lowercase)
    for char in word:
        assert alphabet[char][0].sum() > 0  # observed glyphs are non-blank
    for char in "z":  # a character definitely not in "quick"
        assert any(img.sum() > 0 for img in alphabet[char])

    # Stage 3: text rendering with the generated alphabet.
    renderer = TextRenderer(alphabet, config)
    rendered = renderer.render_text("quick fox")
    assert rendered.ndim == 2
    assert rendered.sum() > 0
    assert rendered.shape[0] > 0 and rendered.shape[1] > 0


def test_end_to_end_handles_repeated_letters(config):
    word = "hello"
    image_bgr = _synthetic_word_image(word)
    analysis = analyze_sample(image_bgr, word, config)
    assert analysis.chars == list(word)
    # 'l' repeats twice; observed_glyphs should still contain exactly one entry for it.
    assert "l" in analysis.observed_glyphs
    alphabet = generate_alphabet_baseline(analysis, config)
    assert len(alphabet) == 26
