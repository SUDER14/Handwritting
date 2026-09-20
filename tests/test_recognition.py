"""Unit tests for pluggable recognition backends (src/recognition/)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recognition import get_recognizer
from src.recognition.cnn_model import CharCNN
from src.recognition.cnn_recognizer import normalize_glyph_for_cnn
from src.recognition.labeled import LabeledRecognizer
from src.utils.config import resolve_path


def test_labeled_recognizer_zips_glyphs_to_label_chars():
    recognizer = LabeledRecognizer(label_text="cat")
    dummy_glyphs = [np.zeros((10, 10), dtype=np.uint8) for _ in range(3)]
    assert recognizer.recognize(dummy_glyphs) == ["c", "a", "t"]


def test_labeled_recognizer_raises_on_length_mismatch():
    recognizer = LabeledRecognizer(label_text="cat")
    dummy_glyphs = [np.zeros((10, 10), dtype=np.uint8) for _ in range(2)]
    with pytest.raises(ValueError):
        recognizer.recognize(dummy_glyphs)


def test_get_recognizer_factory_labeled():
    recognizer = get_recognizer("labeled", label_text="dog")
    assert recognizer.recognize([np.zeros((5, 5), dtype=np.uint8)] * 3) == ["d", "o", "g"]


def test_get_recognizer_factory_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_recognizer("not_a_real_backend")


def test_template_matching_recognizer_returns_one_char_per_glyph():
    recognizer = get_recognizer("template", charset="ab")
    dummy_glyphs = [np.zeros((30, 30), dtype=np.uint8) for _ in range(2)]
    result = recognizer.recognize(dummy_glyphs)
    assert len(result) == 2
    assert all(c in "ab" for c in result)


def test_char_cnn_forward_shape():
    model = CharCNN(num_classes=26)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 26)


def test_normalize_glyph_for_cnn_handles_empty_glyph():
    empty = np.zeros((10, 10), dtype=np.uint8)
    out = normalize_glyph_for_cnn(empty)
    assert out.shape == (28, 28)
    assert out.max() == 0


def test_normalize_glyph_for_cnn_centers_and_resizes():
    glyph = np.zeros((40, 20), dtype=np.uint8)
    glyph[:, :] = 255
    out = normalize_glyph_for_cnn(glyph)
    assert out.shape == (28, 28)
    assert out.max() == 255
    ys, xs = np.nonzero(out)
    # Roughly centered: the ink's bounding box should sit near the canvas middle.
    assert 0 <= xs.min() < 14
    assert xs.max() > 13


CNN_CHECKPOINT = resolve_path("models/checkpoints/char_cnn.pt")


@pytest.mark.skipif(not CNN_CHECKPOINT.exists(), reason="CNN recognizer not trained (run scripts/train_cnn_recognizer.py)")
def test_cnn_recognizer_factory_returns_one_char_per_glyph():
    recognizer = get_recognizer("cnn", checkpoint_path=str(CNN_CHECKPOINT))
    dummy_glyphs = [np.zeros((30, 30), dtype=np.uint8) for _ in range(3)]
    result = recognizer.recognize(dummy_glyphs)
    assert len(result) == 3
    assert all(c in "abcdefghijklmnopqrstuvwxyz" for c in result)
    assert len(recognizer.last_confidences) == 3


def test_cnn_recognizer_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_recognizer("cnn", checkpoint_path=str(tmp_path / "does_not_exist.pt"))
