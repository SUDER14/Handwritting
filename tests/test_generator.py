"""Unit tests for the baseline (non-neural) glyph generator."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.style_extractor import StyleProfile
from src.generator.baseline_generator import BaselineGlyphGenerator


@pytest.fixture
def style() -> StyleProfile:
    return StyleProfile(
        glyph_features=[],
        mean_height_px=60.0,
        mean_width_px=40.0,
        mean_aspect_ratio=0.66,
        mean_slant_deg=10.0,
        slant_std_deg=2.0,
        mean_stroke_width_px=6.0,
        mean_curvature_score=0.5,
        mean_char_spacing_px=8.0,
        baseline_y=100,
        x_height_top_y=40,
        feature_vector=np.zeros(8, dtype=np.float32),
    )


@pytest.fixture
def generator() -> BaselineGlyphGenerator:
    return BaselineGlyphGenerator({"generation": {"variants_per_char": 2, "jitter_rotation_deg": 1.0}})


def test_generate_unobserved_returns_requested_variant_count(generator, style):
    variants = generator.generate_unobserved("a", style)
    assert len(variants) == 2
    for v in variants:
        assert v.ndim == 2
        assert v.sum() > 0  # not a blank image


def test_generate_unobserved_matches_target_height(generator, style):
    variants = generator.generate_unobserved("m", style)
    for v in variants:
        assert abs(v.shape[0] - style.mean_height_px) <= 2


def test_generate_observed_preserves_real_pixels_shape(generator, style):
    real_glyph = np.zeros((80, 50), dtype=np.uint8)
    real_glyph[10:70, 10:40] = 255
    normalized = generator.generate_observed(real_glyph, style)
    assert normalized.shape[0] == int(round(style.mean_height_px))
    assert normalized.sum() > 0


def test_generate_alphabet_uses_observed_where_available(generator, style):
    observed_q = np.zeros((70, 45), dtype=np.uint8)
    observed_q[5:65, 5:40] = 255
    alphabet = generator.generate_alphabet({"q": observed_q}, style, charset="abq")
    assert set(alphabet.keys()) == {"a", "b", "q"}
    assert len(alphabet["q"]) == 1       # observed: single normalized real glyph
    assert len(alphabet["a"]) == 2       # unobserved: variants_per_char
