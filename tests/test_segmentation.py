"""Unit tests for character segmentation (src/segmentation/segmenter.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.segmenter import Segmenter


@pytest.fixture
def segmenter() -> Segmenter:
    return Segmenter()


def make_three_blob_image() -> np.ndarray:
    """A 60x300 binary image (ink=255) with three well-separated square blobs."""
    img = np.zeros((60, 300), dtype=np.uint8)
    img[10:50, 20:60] = 255
    img[10:50, 120:160] = 255
    img[10:50, 220:260] = 255
    return img


def make_dot_and_body_image() -> np.ndarray:
    """One tall body blob with a small dot above it, simulating a lowercase 'i'."""
    img = np.zeros((60, 40), dtype=np.uint8)
    img[20:50, 15:25] = 255  # body
    img[5:10, 17:23] = 255   # dot
    return img


def test_segment_finds_expected_number_of_components(segmenter):
    img = make_three_blob_image()
    result = segmenter.segment(img)
    assert len(result.glyphs) == 3


def test_segment_orders_glyphs_left_to_right(segmenter):
    img = make_three_blob_image()
    result = segmenter.segment(img)
    xs = [g.bbox[0] for g in result.glyphs]
    assert xs == sorted(xs)


def test_diacritic_merge_combines_dot_and_body(segmenter):
    img = make_dot_and_body_image()
    result = segmenter.segment(img)
    assert len(result.glyphs) == 1
    # Merged bbox should span from the dot's top to the body's bottom.
    _, y, _, h = result.glyphs[0].bbox
    assert y <= 10
    assert y + h >= 45


def test_segment_and_align_merges_extra_components(segmenter):
    img = make_three_blob_image()
    result = segmenter.segment_and_align_to_label(img, "ab")  # only 2 target chars
    assert len(result.glyphs) == 2


def test_segment_and_align_splits_when_too_few_components(segmenter):
    img = make_three_blob_image()
    result = segmenter.segment_and_align_to_label(img, "abcd")  # 4 target chars, 3 found
    assert len(result.glyphs) == 4


def test_empty_image_returns_no_glyphs(segmenter):
    img = np.zeros((60, 300), dtype=np.uint8)
    result = segmenter.segment(img)
    assert len(result.glyphs) == 0


def make_touching_pair_image() -> np.ndarray:
    """Two ~equal-width blobs joined by a thin bridge (simulates a cursive join)
    alongside one normal, separate blob — the bridge keeps them one connected
    component, but the bridge is much thinner than the blobs themselves, so
    the vertical ink-density profile should still show a clear valley there.
    """
    img = np.zeros((60, 220), dtype=np.uint8)
    img[10:50, 10:40] = 255                 # normal separate glyph, width 30
    img[10:50, 90:120] = 255                # touching glyph, part A, width 30
    img[25:35, 120:130] = 255               # thin bridge (10px wide, only 10px tall)
    img[10:50, 130:160] = 255               # touching glyph, part B, width 30
    return img


def test_split_touching_glyphs_separates_cursive_join(segmenter):
    img = make_touching_pair_image()
    result = segmenter.segment(img)
    # Without splitting this would be 2 components (the normal glyph + one
    # wide joined blob); with splitting it should recover 3 glyphs.
    assert len(result.glyphs) == 3
    xs = [g.bbox[0] for g in result.glyphs]
    assert xs == sorted(xs)


def test_split_touching_glyphs_can_be_disabled_via_config():
    segmenter = Segmenter({"segmentation": {
        "min_component_area": 20, "merge_gap_px": 4, "split_touching_glyphs": False,
    }})
    img = make_touching_pair_image()
    result = segmenter.segment(img)
    assert len(result.glyphs) == 2
