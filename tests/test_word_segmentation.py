"""Task 4.1: projection-profile word segmentation with a per-line gap rule learned from word counts."""
import numpy as np

from src.segmentation.words import (canonical, column_gaps, despeckle, learn_gap_rule, predicted_word_count,
                                    references_from_lines, segment_words, shear, slant_angle)

RULE = {"floor": 11.0, "rel_max": 0.4}


def _line(words, letter_gap=4, word_gap=30, h=64, letter_w=10):
    """Canonical ink (255 on 0): `words` = letters per word; letters separated by letter_gap, words by word_gap."""
    w = 10 + sum(n * letter_w + (n - 1) * letter_gap for n in words) + (len(words) - 1) * word_gap + 10
    img = np.zeros((h, w), np.uint8)
    x = 10
    for n in words:
        for i in range(n):
            img[20:44, x:x + letter_w] = 255
            x += letter_w + (letter_gap if i < n - 1 else 0)
        x += word_gap
    return img


def test_gaps_are_empty_column_runs_inside_the_ink_span_with_tolerance():
    img = _line([2, 1], letter_gap=4, word_gap=30)
    gaps, x0, x1 = column_gaps(img)
    assert (x0, x1) == (10, 10 + 24 + 30 + 10) and [g[0] for g in gaps] == [4, 30]
    img[40:42, 10 + 24 + 5] = 255                      # a 2-px descender tail inside the word gap
    assert [g[0] for g in column_gaps(img)[0]] == [4, 5, 24]
    assert [g[0] for g in column_gaps(img, col_tol=2)[0]] == [4, 30]


def test_per_line_rule_handles_a_tight_spacer_and_a_wide_spacer():
    wide, tight = _line([3, 2, 4], letter_gap=6, word_gap=40), _line([3, 2, 4], letter_gap=3, word_gap=12)
    assert predicted_word_count(wide, RULE) == 3 and predicted_word_count(tight, RULE) == 3
    fit = learn_gap_rule([(wide, 3), (tight, 3), (_line([2, 2, 2, 2]), 4), (_line([5]), 1)])
    assert fit["within1_acc"] == 1.0 and fit["exact_acc"] == 1.0


def test_deslant_recovers_the_shear_of_slanted_bars():
    img = _line([1, 1, 1], word_gap=20)
    sheared, _ = shear(img, 30.0)
    corr = slant_angle(sheared)
    assert abs(corr + 30.0) <= 2.5 and slant_angle(img) == 0.0         # correcting angle
    assert abs(slant_angle(shear(sheared, corr)[0])) <= 2.5                 # applying it leaves the bars upright


def test_despeckle_removes_dust_that_would_split_a_word_gap():
    img = _line([2, 2], word_gap=30)
    img[30, 10 + 24 + 15] = 255
    assert [g[0] for g in column_gaps(despeckle(img))[0]] == [4, 30, 4]


def test_segment_words_returns_native_resolution_crops_left_to_right():
    native = 255 - np.kron(_line([2, 3]), np.ones((4, 4), np.uint8))       # 4x native scale, dark ink on light
    crops = segment_words(native, RULE)
    assert len(crops) == 2 and crops[0].shape[1] < crops[1].shape[1] and crops[0].shape[0] == 24 * 4
    assert [r.shape[0] for r in references_from_lines([native], RULE, height=32)] == [32, 32]
    assert canonical(native).shape[0] == 64


def test_segment_words_on_slanted_writing_keeps_each_word_whole():
    ink = _line([2, 3, 2], word_gap=14)
    slanted, _ = shear(ink, 35.0)                       # neighbouring words overlap in columns before deslanting
    assert predicted_word_count(slanted, RULE) == 3
    crops = segment_words(255 - np.kron(slanted, np.ones((2, 2), np.uint8)), RULE)
    assert len(crops) == 3
