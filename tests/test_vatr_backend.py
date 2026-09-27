"""Task 3.3 adaptation: reference lines -> VATr word crops (no VATr checkpoint needed for these)."""
import numpy as np

from src.evaluation.protocol import Ref
from src.generator.vatr_backend import (NUM_STYLE, STYLE_H, STYLE_MAX_W, cycle_to, reference_word_crops,
                                        split_line_into_words, style_tensor_array, transcription_words)


def _line(blocks, h=60, w=400):
    """Paper 255 with dark rectangles at the given (x0, x1, y0, y1)."""
    img = np.full((h, w), 255, np.uint8)
    for x0, x1, y0, y1 in blocks:
        img[y0:y1, x0:x1] = 0
    return img


def test_transcription_words_attaches_punctuation():
    assert transcription_words("and State housing projects .") == ["and", "State", "housing", "projects."]
    assert transcription_words("M Ps tomorrow. Mr. Michael") == ["M", "Ps", "tomorrow.", "Mr.", "Michael"]


def test_split_uses_the_widest_gaps_only():
    # three words; inside word 2 there is a narrow 3-px gap (between letters) that must NOT be cut
    img = _line([(10, 60, 20, 40), (100, 130, 15, 45), (133, 160, 20, 40), (220, 300, 10, 50)])
    crops = split_line_into_words(img, 3)
    assert [c.shape[1] for c in crops] == [50, 60, 80]
    assert crops[1].shape[0] == 30                                  # trimmed to its own ink box (15..45)


def test_split_never_invents_cuts_inside_ink():
    img = _line([(10, 200, 20, 40)])
    assert len(split_line_into_words(img, 4)) == 1


def test_reference_crops_cycle_to_fifteen_style_slots():
    refs = [Ref(_line([(10, 60, 20, 40), (100, 160, 20, 40)]), "ab cd")]
    crops = reference_word_crops(refs)
    assert len(crops) == 2
    slots = cycle_to(crops, NUM_STYLE)
    assert len(slots) == NUM_STYLE and slots[2] is crops[0]


def test_style_preprocessing_matches_vatr_folder_dataset():
    word = np.full((20, 40), 255, np.uint8)
    word[5:15, 5:35] = 0
    arr = style_tensor_array([word, np.full((10, 500), 0, np.uint8)])
    assert arr.shape == (2, STYLE_H, STYLE_MAX_W)
    assert arr.min() >= -1.0 and arr.max() <= 1.0
    assert np.all(arr[0][:, 64:] == 1.0)                            # right padding is paper (+1 after normalisation)
    assert np.all(arr[1] == -1.0)                                   # over-wide word cropped to 192, all ink
