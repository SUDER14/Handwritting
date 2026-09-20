"""GlyphMetrics: conventions, estimation, and what the renderer does with them."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from src.features.line_metrics import estimate_line_metrics
from src.generator.alphabet_builder import AlphabetBuilder, AlphabetMetadata
from src.generator.baseline_generator import BaselineGlyphGenerator
from src.generator.glyph import (
    GlyphBitmap, GlyphMetrics, GlyphPriors, bitmap_from_generated, glyph_class, make_bitmap,
)
from src.pipeline import analyze_sample, generate_alphabet_baseline, generate_alphabet_neural
from src.renderer.text_renderer import TextRenderer
from src.utils.config import PROJECT_ROOT, load_config, resolve_path

TARGET = 40
CFG = {"renderer": {"target_x_height_px": TARGET, "char_spacing_px": 0, "word_spacing_px": 20,
                    "line_height_multiplier": 1.0, "baseline_jitter_px": 0, "rotation_jitter_deg": 0.0,
                    "canvas_margin_px": 10}}


def block(h: int, w: int = 20) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


def ink_rows(img: np.ndarray, x0: int = 0, x1: int | None = None) -> tuple[int, int]:
    rows = np.nonzero(img[:, x0:x1].max(axis=1))[0]
    return int(rows[0]), int(rows[-1])


# -- the dataclass ---------------------------------------------------------------

def test_metrics_sign_convention_baseline_offset():
    on_line = make_bitmap(block(30), x_height=30, baseline_row=30, spacing=4)       # bottom on the baseline
    descender = make_bitmap(block(50), x_height=30, baseline_row=30, spacing=4)     # baseline 30 rows down a 50-row glyph
    assert on_line.metrics.baseline_offset == 0
    assert descender.metrics.baseline_offset == -20                                 # negative: 20 px below the line
    assert descender.metrics.bbox[1] == -30 and descender.metrics.bbox[1] + descender.metrics.bbox[3] == 20


def test_metrics_validate_consistency_and_scale():
    with pytest.raises(ValueError, match="inconsistent"):
        GlyphMetrics(advance_width=10, x_height=5, baseline_offset=-3, bbox=(0, -10, 8, 10))   # bottom is at 0, not 3
    with pytest.raises(ValueError):
        GlyphMetrics(advance_width=10, x_height=0, baseline_offset=0, bbox=(0, -5, 8, 5))
    m = make_bitmap(block(50, 20), 30, 30, 6).metrics.scaled(2.0)
    assert (m.x_height, m.baseline_offset, m.advance_width) == (60, -40, 52)
    assert m.bbox == (6.0, -60.0, 40.0, 100.0)
    assert GlyphMetrics.from_dict(json.loads(json.dumps(m.to_dict()))) == m


def test_glyph_bitmap_rejects_metrics_that_do_not_match_the_image():
    with pytest.raises(ValueError, match="does not match"):
        GlyphBitmap(block(30, 20), make_bitmap(block(40, 20), 30, 40, 0).metrics)


# -- estimation ---------------------------------------------------------------------

def _boxes(spec):  # [(char, y_top, height)] -> chars, glyph-like objects
    return [c for c, _, _ in spec], [SimpleNamespace(bbox=(10 * i, y, 8, h)) for i, (_, y, h) in enumerate(spec)]


def test_line_metrics_use_x_height_letters_and_ignore_ascenders_and_descenders():
    chars, glyphs = _boxes([("q", 60, 121), ("u", 64, 72), ("i", 60, 74), ("c", 50, 80), ("k", 10, 118)])
    lm = estimate_line_metrics(chars, glyphs)
    assert lm.source == "x_height_letters" and lm.n_used == 2          # u and c only
    assert lm.x_height == 76 and lm.baseline_y == 133                    # medians of (72, 80) and (136, 130)


def test_line_metrics_fall_back_to_priors_when_no_x_height_letter_is_present():
    p = GlyphPriors()
    chars, glyphs = _boxes([("f", 0, 100), ("l", 0, 100)])              # two ascender letters: height 100 = 2.0 x-heights
    lm = estimate_line_metrics(chars, glyphs, p)
    assert lm.source == "priors" and lm.x_height == pytest.approx(100 / p.ascender_ratio)
    assert lm.baseline_y == 100                                          # they stand on the line


def test_generated_glyph_metrics_come_from_priors_and_put_descenders_below_the_line():
    p = GlyphPriors()
    a, k, g, i = (bitmap_from_generated(c, np.pad(block(20), 4), 0.2, p) for c in "akgi")
    assert glyph_class("g") == "descender" and glyph_class("k") == "ascender"
    assert a.metrics.x_height == 20 and a.metrics.baseline_offset == 0
    assert k.metrics.x_height == pytest.approx(20 / p.ascender_ratio) and k.metrics.baseline_offset == 0
    assert g.metrics.baseline_offset < 0 and g.metrics.baseline_offset == pytest.approx(-p.descender_depth_ratio * g.metrics.x_height)
    assert i.metrics.x_height == pytest.approx(20 / p.dotted_ratio)
    assert a.image.shape == (20, 20)                                     # padding cropped away: tight to the ink


# -- baseline generator: real geometry from the font ------------------------------------

def _style(x_height=40.0):
    from src.features.style_extractor import StyleProfile
    return StyleProfile([], 60, 40, 0.66, 0.0, 0.0, 6.0, 0.5, 6.0, 100, 60, np.zeros(8, np.float32), x_height_px=x_height)


@pytest.fixture(scope="module")
def baseline_alphabet():
    gen = BaselineGlyphGenerator({"generation": {"variants_per_char": 1, "jitter_rotation_deg": 0.0}})
    return {c: gen.generate_unobserved(c, _style())[0] for c in "abcdefghijklmnopqrstuvwxyz"}


def test_descenders_have_negative_baseline_offset_and_others_do_not(baseline_alphabet):
    xh = baseline_alphabet["x"].metrics.x_height
    for c in "gjpqy":
        assert baseline_alphabet[c].metrics.baseline_offset < -0.4 * xh, c         # real descender: ~ -0.56 x-heights
    for c in "acemnorsuvwzbdhkl":
        # The reference font is a cursive script, so exit strokes dip a little below the line (up to ~0.25 x-heights
        # for 'x'); the point is that none is anywhere near a true descender.
        assert -0.3 * xh < baseline_alphabet[c].metrics.baseline_offset < 0.2 * xh, c


def test_ascenders_are_taller_than_the_x_height_band(baseline_alphabet):
    xh = baseline_alphabet["x"].metrics.x_height
    for c in "bdhkl":
        assert -baseline_alphabet[c].metrics.bbox[1] > 1.4 * xh, c
    assert -baseline_alphabet["a"].metrics.bbox[1] < 1.25 * xh


# -- renderer: common x-height, baseline placement ---------------------------------------

def test_descender_drops_below_the_baseline_while_neighbours_sit_on_it():
    xh = 30.0
    alphabet = {
        "a": [make_bitmap(block(30), xh, 30, 0)],                         # x-height letter on the line
        "g": [make_bitmap(block(51), xh, 30, 0)],                         # 21 px (0.7 x-heights) below the line
    }
    renderer = TextRenderer(alphabet, CFG)
    img = renderer.render_line("ag")
    per_glyph = renderer.mean_advance                                     # both glyphs advance by 20*k
    k = TARGET / xh
    split = 10 + int(round(20 * k))                                       # x where 'g' starts (margin + a's advance)
    a_top, a_bottom = ink_rows(img, 0, split)
    g_top, g_bottom = ink_rows(img, split)
    baseline_y = 10 + renderer.ascent                                     # margin + ascent (multiplier 1.0 -> no leading)
    assert a_bottom + 1 == baseline_y                                     # 'a' bottom edge is exactly on the baseline
    assert g_bottom + 1 - baseline_y == pytest.approx(21 * k, abs=1)      # 'g' hangs 21*k px below it
    assert g_top == a_top                                                 # both tops at the x-height


def test_descender_is_not_clipped_by_the_canvas():
    alphabet = {"g": [make_bitmap(block(60), 30, 30, 0)]}
    img = TextRenderer(alphabet, CFG).render_line("g")
    top, bottom = ink_rows(img)
    assert bottom < img.shape[0] - 1 and img[bottom:, :].sum() > 0        # ink ends inside the canvas
    assert img[-CFG["renderer"]["canvas_margin_px"] // 2:, :].sum() == 0  # margin below the tail is blank


def test_glyphs_from_28px_and_100px_sources_render_at_the_same_x_height():
    """The mismatch this fixes: VAE output ~28 px vs real crops ~100 px. Metrics say what each pixel size means."""
    alphabet = {
        "a": [make_bitmap(block(28, 24), x_height=28, baseline_row=28, spacing=0)],      # 'generated', tiny
        "b": [make_bitmap(block(100, 80), x_height=100, baseline_row=100, spacing=0)],   # 'observed', big
    }
    img = TextRenderer(alphabet, CFG).render_line("ab")
    heights = {}
    for name, cols in (("a", slice(0, 10 + 32)), ("b", slice(10 + 34, None))):       # a is 24*40/28 = 34 px wide
        rows = np.nonzero(img[:, cols].max(axis=1))[0]
        heights[name] = rows[-1] - rows[0] + 1
    assert heights["a"] == heights["b"] == TARGET


def test_narrow_and_wide_glyphs_keep_their_proportions():
    """Common x-height, NOT common bounding box: widths still differ after normalisation."""
    alphabet = {"i": [make_bitmap(block(30, 6), 30, 30, 0)], "m": [make_bitmap(block(30, 45), 30, 30, 0)]}
    r = TextRenderer(alphabet, CFG)
    wi, wm = r.render_line("i").shape[1], r.render_line("m").shape[1]
    assert wm - wi == pytest.approx(round((45 - 6) * TARGET / 30), abs=2)


def test_advance_width_controls_spacing():
    tight = {"a": [make_bitmap(block(30, 20), 30, 30, spacing=0)]}
    loose = {"a": [make_bitmap(block(30, 20), 30, 30, spacing=30)]}
    assert TextRenderer(loose, CFG).render_line("aaa").shape[1] - TextRenderer(tight, CFG).render_line("aaa").shape[1] \
        == pytest.approx(3 * 30 * TARGET / 30, abs=3)


# -- integration on the committed sample ----------------------------------------------------

@pytest.fixture(scope="module")
def quick_analysis():
    img = cv2.imread(str(resolve_path("data/samples/quick_sample.png")), cv2.IMREAD_COLOR)
    return analyze_sample(img, "quick", load_config())


def test_observed_glyphs_get_metrics_measured_from_the_sample(quick_analysis):
    bm = quick_analysis.observed_bitmaps
    assert set(bm) == set("quick")
    xh = {g.metrics.x_height for g in bm.values()}
    assert len(xh) == 1 and xh.pop() > 20                                 # one writer x-height for the whole line
    assert bm["q"].metrics.baseline_offset < -0.3 * bm["u"].metrics.x_height   # q's tail drops below the line
    for c in "uc":
        assert abs(bm[c].metrics.baseline_offset) <= 4                    # x-height letters stand on it
    assert bm["k"].metrics.bbox[1] < -1.2 * bm["k"].metrics.x_height      # ascender rises above the x-height band
    assert quick_analysis.style_profile.line_metrics.source == "x_height_letters"
    for g in bm.values():                                                 # metrics agree with the crops themselves
        assert g.metrics.bbox[2] == g.image.shape[1] and g.metrics.bbox[3] == g.image.shape[0]


def test_baseline_alphabet_renders_q_with_a_tail_below_the_line(quick_analysis):
    cfg = load_config()
    alphabet = generate_alphabet_baseline(quick_analysis, cfg)
    r = TextRenderer(alphabet, {**cfg, "renderer": {**cfg["renderer"], "baseline_jitter_px": 0, "rotation_jitter_deg": 0.0}})
    img = r.render_line("nq")
    baseline_y = cfg["renderer"]["canvas_margin_px"] + r.ascent + int((r.ascent + r.descent) * (cfg["renderer"]["line_height_multiplier"] - 1) // 2)
    xs = np.nonzero(img.max(axis=0))[0]
    cut = (xs.min() + xs.max()) // 2
    assert ink_rows(img, 0, cut)[1] + 1 <= baseline_y + 3                 # 'n' rests on the line
    assert ink_rows(img, cut)[1] + 1 > baseline_y + 0.3 * cfg["renderer"]["target_x_height_px"]  # 'q' descends


CNN_CKPT, VAE_CKPT = resolve_path("models/checkpoints/char_cnn.pt"), resolve_path("models/checkpoints/style_vae.pt")


@pytest.mark.skipif(not VAE_CKPT.exists(), reason="needs a trained VAE checkpoint")
def test_neural_alphabet_has_consistent_metrics_across_28px_and_native_glyphs(quick_analysis):
    cfg = load_config()
    alphabet = generate_alphabet_neural(quick_analysis, checkpoint_path=str(VAE_CKPT), config=cfg)
    native = {c for c in "quick"}
    sizes = {c: alphabet[c][0].image.shape for c in alphabet}
    assert max(sizes[c][0] for c in native) > 60 and sizes["a"][0] <= 28   # the raw mismatch is still there...
    r = TextRenderer(alphabet, cfg)
    for c in "aemnors":                                                    # ...but generated x-height letters normalise to one height
        g = alphabet[c][0]                                                 # (observed ones vary with their own measured height)
        assert round(g.image.shape[0] * r._scale(g)) == pytest.approx(cfg["renderer"]["target_x_height_px"], abs=1), c
    for c in "gpqy":
        assert alphabet[c][0].metrics.baseline_offset < 0


# -- persistence -----------------------------------------------------------------------------

def test_alphabet_builder_round_trips_metrics_and_rejects_legacy_alphabets(tmp_path):
    alphabet = {"a": [make_bitmap(block(30, 20), 30, 30, 4), make_bitmap(block(31, 21), 30, 30, 4)],
                "g": [make_bitmap(block(50, 20), 30, 30, 4)]}
    builder = AlphabetBuilder(tmp_path)
    builder.save(alphabet, AlphabetMetadata("baseline", ["a"], ["g"]), "s1")
    loaded, meta = builder.load("s1")
    assert set(loaded) == {"a", "g"} and len(loaded["a"]) == 2
    for c in alphabet:
        for before, after in zip(alphabet[c], loaded[c]):
            assert after.metrics == before.metrics and np.array_equal(after.image, before.image)
    assert loaded["g"][0].metrics.baseline_offset == -20 and meta["glyphs"]["g"]["metrics"][0]["baseline_offset"] == -20

    meta_path = tmp_path / "s1" / "glyph_metadata.json"
    legacy = json.loads(meta_path.read_text())
    for entry in legacy["glyphs"].values():
        entry.pop("metrics")
    meta_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="before glyph metrics existed"):
        builder.load("s1")


# -- config hygiene ------------------------------------------------------------------------------

def _leaf_keys(node, out=None):
    out = set() if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            (_leaf_keys(v, out) if isinstance(v, dict) else out.add(k))
    return out


def test_no_config_key_is_read_nowhere():
    """Guards against dead knobs (7 were found unread in the audit): every leaf key in config.yaml
    must at least be named somewhere in src/, scripts/ or app/. Heuristic: catches unused, not misused."""
    code = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for d in ("src", "scripts", "app") for p in (PROJECT_ROOT / d).rglob("*.py"))
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    unread = sorted(k for k in _leaf_keys(cfg) if not re.search(rf"\b{re.escape(k)}\b", code))
    assert not unread, f"config keys not referenced by any code: {unread}"
