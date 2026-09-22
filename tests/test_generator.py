"""Unit tests for the baseline (non-neural) glyph generator."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.style_extractor import StyleProfile
from src.generator.baseline_generator import BaselineGlyphGenerator
from src.generator.glyph import GlyphBitmap, make_bitmap


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
        x_height_top_y=60,
        feature_vector=np.zeros(8, dtype=np.float32),
        x_height_px=40.0,
    )


@pytest.fixture
def generator() -> BaselineGlyphGenerator:
    return BaselineGlyphGenerator({"generation": {"variants_per_char": 2, "jitter_rotation_deg": 1.0}})


def test_generate_unobserved_returns_requested_variant_count(generator, style):
    variants = generator.generate_unobserved("a", style)
    assert len(variants) == 2
    for v in variants:
        assert isinstance(v, GlyphBitmap)
        assert v.image.ndim == 2
        assert v.image.sum() > 0  # not a blank image


def test_generated_glyphs_are_not_forced_into_one_box(generator, style):
    """The old generator resized every letter to the same (mean_height x mean_width) box. Now each keeps
    its natural proportions: 'i' narrow, 'm' wide, 'k' taller than 'a', 'g' deeper than 'a'."""
    # Proportions only: give the writer the font's own slant so the (writer - font) shear is zero and bounding-box
    # widths are not inflated by a shear that widens tall narrow letters more than short wide ones.
    from dataclasses import replace
    upright = replace(style, mean_slant_deg=generator._font_slant())
    g = {c: generator.generate_unobserved(c, upright)[0] for c in "imakg"}
    assert g["m"].image.shape[1] > 2 * g["i"].image.shape[1]
    assert g["k"].image.shape[0] > 1.4 * g["a"].image.shape[0]
    assert len({v.image.shape for v in g.values()}) == len(g)      # no two letters share a box


def test_x_height_is_consistent_across_letters_in_font_pixels(generator, style):
    """All font-derived glyphs share one x-height (their bitmaps are at the same font scale)."""
    xs = {c: generator.generate_unobserved(c, style)[0].metrics.x_height for c in "amkgq"}
    assert max(xs.values()) == pytest.approx(min(xs.values()))


def test_generate_unobserved_requires_an_x_height(generator, style):
    style.x_height_px = 0.0
    with pytest.raises(ValueError, match="x_height_px"):
        generator.generate_unobserved("a", style)


def test_generate_alphabet_uses_observed_where_available(generator, style):
    q_image = np.zeros((70, 45), dtype=np.uint8)
    q_image[5:65, 5:40] = 255
    observed_q = make_bitmap(q_image, x_height=40.0, baseline_row=50.0, spacing=6.0)
    alphabet = generator.generate_alphabet({"q": observed_q}, style, charset="abq")
    assert set(alphabet.keys()) == {"a", "b", "q"}
    assert len(alphabet["q"]) == 1 and alphabet["q"][0] is observed_q   # observed: the real glyph, untouched
    assert len(alphabet["a"]) == 2                                      # unobserved: variants_per_char


# -- ROADMAP 2.1: shear = writer slant - font slant, in the estimator's own sign convention ------------------------

def test_shearing_a_vertical_bar_by_plus_20_is_read_back_as_plus_20_by_the_slant_estimator():
    from src.features.style_extractor import StyleFeatureExtractor
    from src.generator.baseline_generator import _apply_shear
    ex = StyleFeatureExtractor()
    bar = np.zeros((120, 60), np.uint8)
    bar[10:110, 26:34] = 255
    for angle in (20.0, -20.0, 8.0):
        assert ex._slant_deg(_apply_shear(bar, angle)) == pytest.approx(angle, abs=1.0)
    assert np.array_equal(_apply_shear(bar, 0.0), bar)


def test_font_slant_is_measured_by_our_estimator_and_cached(tmp_path):
    from src.generator import baseline_generator as bg
    font = bg._load_reference_font()
    bg._FONT_SLANT_CACHE.clear()
    v1 = bg.font_slant_deg(font, cache_dir=str(tmp_path))
    assert (tmp_path / "font_slant.json").exists()
    bg._FONT_SLANT_CACHE.clear()
    assert bg.font_slant_deg(font, cache_dir=str(tmp_path)) == v1              # read back from the disk cache
    assert -90 < v1 < 90


def test_generator_shears_by_writer_slant_minus_font_slant(monkeypatch, style):
    """A writer whose slant equals the font's own gets NO shear; +/- 15 deg relative to the font shears by exactly that."""
    from src.generator import baseline_generator as bg
    seen = []
    monkeypatch.setattr(bg, "_apply_shear", lambda canvas, deg: (seen.append(deg), canvas)[1])
    gen = BaselineGlyphGenerator({})
    monkeypatch.setattr(gen, "_font_slant", lambda: 22.0)
    from dataclasses import replace
    for writer_slant, expected in ((22.0, 0.0), (37.0, 15.0), (7.0, -15.0)):
        seen.clear()
        gen.generate_unobserved("k", replace(style, mean_slant_deg=writer_slant))
        assert seen and all(d == pytest.approx(expected) for d in seen)
