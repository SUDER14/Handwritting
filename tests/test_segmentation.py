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


# -- stacked-component merge: the dot of 'i' rejoins its stem ------------------------------------
import cv2  # noqa: E402

from src.preprocessing.pipeline import Preprocessor  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402


def _glyph_count(image_path: str, n_chars: int) -> int:
    img = cv2.imread(str(resolve_path(image_path)), cv2.IMREAD_COLOR)
    cropped = Preprocessor().process(image_bgr=img, n_chars=n_chars).cropped
    return len(Segmenter().segment(cropped).glyphs)      # segment(), NOT the label-aligned variant


def test_quick_segments_to_exactly_five_glyphs():
    """q-u-i-c-k is 5 letters (6 connected components: the i-dot is separate). It used to come out as 6
    because a wrong deskew sheared the dot off its stem, so the dot never overlapped the stem in x."""
    assert _glyph_count("data/samples/quick_sample.png", n_chars=5) == 5


def test_hello_segments_to_exactly_five_glyphs():
    assert _glyph_count("data/samples/hello_sample.png", n_chars=5) == 5


def test_dot_merges_with_stem_regardless_of_dot_size(segmenter):
    """Not just tiny dots: a big dot (taller than half the stem) still merges when it sits over the stem."""
    img = np.zeros((90, 60), dtype=np.uint8)
    img[40:80, 20:30] = 255      # stem, 10 wide
    img[18:32, 22:32] = 255      # dot: 14 tall (> half the median height), 8 of 10 px over the stem
    result = segmenter.segment(img)
    assert len(result.glyphs) == 1
    _, y, _, h = result.glyphs[0].bbox
    assert y == 18 and y + h == 80


def test_offset_dot_without_x_overlap_is_not_merged(segmenter):
    img = np.zeros((90, 80), dtype=np.uint8)
    img[40:80, 20:30] = 255      # stem
    img[18:30, 50:58] = 255      # dot far to the right: no x-overlap
    assert len(segmenter.segment(img).glyphs) == 2


def test_distant_stacked_blobs_are_not_merged(segmenter):
    """x-overlap alone is not enough: the vertical gap must be small relative to the glyph height."""
    img = np.zeros((300, 60), dtype=np.uint8)
    img[10:30, 20:40] = 255
    img[250:290, 20:40] = 255    # 220 px away
    assert len(segmenter.segment(img).glyphs) == 2


def test_side_by_side_slanted_letters_with_overlapping_boxes_stay_separate(segmenter):
    """Two parallel slanted strokes whose bounding boxes overlap in x by >50% are neighbours, not a stack:
    their vertical ranges coincide."""
    img = np.zeros((80, 120), dtype=np.uint8)
    cv2.line(img, (10, 60), (40, 15), 255, 5)
    cv2.line(img, (28, 60), (58, 15), 255, 5)
    boxes = [cv2.boundingRect(c) for c in cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]]
    assert len(boxes) == 2 and min(b[0] + b[2] for b in boxes) > max(b[0] for b in boxes)   # boxes do overlap in x
    assert len(segmenter.segment(img).glyphs) == 2


def test_dot_stem_and_descender_merge_transitively(segmenter):
    """A 'j'-like shape whose three parts are separate components collapses into one glyph."""
    img = np.zeros((140, 50), dtype=np.uint8)
    img[10:22, 20:30] = 255      # dot
    img[34:90, 20:30] = 255      # stem
    img[96:130, 16:30] = 255     # detached descender hook
    assert len(segmenter.segment(img).glyphs) == 1
