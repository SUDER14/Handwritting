"""Task 4.1: projection-profile word segmentation with a threshold learned from word counts."""
import numpy as np

from src.segmentation.words import (canonical, column_gaps, despeckle, learn_gap_threshold, predicted_word_count,
                                    references_from_lines, segment_words)


def _line(words, letter_gap=4, word_gap=30, h=64, letter_w=10):
    """Canonical ink (255 on 0): `words` = letters per word; letters separated by letter_gap, words by word_gap."""
    w = 10 + sum(n * letter_w + (n - 1) * letter_gap for n in words) + (len(words) - 1) * word_gap + 10
    img = np.zeros((h, w), np.uint8)
    x = 10
    for k, n in enumerate(words):
        for i in range(n):
            img[20:44, x:x + letter_w] = 255
            x += letter_w + (letter_gap if i < n - 1 else 0)
        x += word_gap
    return img


def test_gaps_are_ink_free_runs_inside_the_ink_span():
    gaps, x0, x1 = column_gaps(_line([2, 1], letter_gap=4, word_gap=30))
    assert (x0, x1) == (10, 10 + 24 + 30 + 10) and [g[0] for g in gaps] == [4, 30]


def test_learned_threshold_separates_letter_gaps_from_word_gaps():
    lines = [(_line(ws), len(ws)) for ws in ([3, 2, 4], [1, 5], [2, 2, 2, 2], [6], [3, 3, 1])]
    fit = learn_gap_threshold(lines)
    assert fit["exact_acc"] == 1.0 and 5 <= fit["threshold"] <= 30
    assert predicted_word_count(_line([2, 3, 2, 1]), fit["threshold"]) == 4


def test_despeckle_removes_dust_that_would_split_a_word_gap():
    img = _line([2, 2], word_gap=30)
    img[30, 10 + 24 + 15] = 255                      # one dust pixel in the middle of the word gap
    assert [g[0] for g in column_gaps(img)[0]].count(30) == 0
    assert [g[0] for g in column_gaps(despeckle(img))[0]] == [4, 30, 4]


def test_segment_words_returns_native_resolution_crops_left_to_right():
    native = 255 - np.kron(_line([2, 3]), np.ones((4, 4), np.uint8))       # 4x native scale, dark ink on light
    crops = segment_words(native, threshold=12)
    assert len(crops) == 2 and crops[0].shape[1] < crops[1].shape[1] and crops[0].shape[0] == 24 * 4
    refs = references_from_lines([native], threshold=12, height=32)
    assert [r.shape[0] for r in refs] == [32, 32]
    assert canonical(native).shape[0] == 64
