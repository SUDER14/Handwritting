"""Unit tests for rule-based style feature extraction (src/features/style_extractor.py)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features.style_extractor import StyleFeatureExtractor
from src.segmentation.segmenter import Glyph


@pytest.fixture
def extractor() -> StyleFeatureExtractor:
    return StyleFeatureExtractor()


def make_circle_glyph(radius: int = 20) -> np.ndarray:
    size = radius * 2 + 10
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, 255, thickness=4)
    return img


def make_thick_vertical_bar(thickness: int = 12, height: int = 60) -> np.ndarray:
    img = np.zeros((height, thickness + 10), dtype=np.uint8)
    img[:, 5:5 + thickness] = 255
    return img


def test_glyph_features_basic_dimensions(extractor):
    glyph = make_thick_vertical_bar()
    feats = extractor.extract_glyph_features("i", glyph)
    assert feats.height_px == glyph.shape[0]
    assert feats.width_px == glyph.shape[1]
    assert feats.char == "i"


def test_stroke_width_scales_with_actual_thickness(extractor):
    thin = make_thick_vertical_bar(thickness=4)
    thick = make_thick_vertical_bar(thickness=16)
    thin_feats = extractor.extract_glyph_features("i", thin)
    thick_feats = extractor.extract_glyph_features("i", thick)
    assert thick_feats.stroke_width_px > thin_feats.stroke_width_px


def test_curvature_score_higher_for_round_shape(extractor):
    circle = make_circle_glyph()
    bar = make_thick_vertical_bar()
    circle_feats = extractor.extract_glyph_features("o", circle)
    bar_feats = extractor.extract_glyph_features("l", bar)
    assert circle_feats.curvature_score > bar_feats.curvature_score


def test_empty_glyph_does_not_crash(extractor):
    empty = np.zeros((30, 30), dtype=np.uint8)
    feats = extractor.extract_glyph_features("x", empty)
    assert feats.ink_density == 0.0


def test_style_profile_aggregates_multiple_glyphs(extractor):
    glyphs = [
        Glyph(index=0, bbox=(0, 0, 14, 60), image=make_thick_vertical_bar(), centroid=(7, 30)),
        Glyph(index=1, bbox=(30, 0, 50, 50), image=make_circle_glyph(), centroid=(55, 25)),
    ]
    profile = extractor.extract_style_profile(["i", "o"], glyphs, baseline_y=60, x_height_top_y=0)
    assert len(profile.glyph_features) == 2
    assert profile.feature_vector.shape == (8,)
    assert profile.mean_height_px > 0
