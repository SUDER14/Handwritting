"""Unit tests for the preprocessing pipeline (src/preprocessing/pipeline.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.pipeline import Preprocessor


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor()


def make_test_image(text_present: bool = True) -> np.ndarray:
    """A simple synthetic BGR image with (optionally) a black rectangle 'ink' blob."""
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    if text_present:
        img[80:120, 150:250] = 0
    return img


def test_load_image_from_array_resizes_to_target_height(preprocessor):
    img = make_test_image()
    result = preprocessor.load_image_from_array(img)
    assert result.shape[0] == preprocessor.cfg["target_line_height"]


def test_to_grayscale_shape(preprocessor):
    img = make_test_image()
    gray = preprocessor.to_grayscale(img)
    assert gray.ndim == 2
    assert gray.shape[:2] == img.shape[:2]


def test_binarize_produces_ink_on_black_background(preprocessor):
    img = make_test_image()
    gray = preprocessor.to_grayscale(img)
    binary = preprocessor.binarize(gray)
    assert set(np.unique(binary)).issubset({0, 255})
    # There should be some foreground (ink) pixels where the black rectangle was.
    assert binary[80:120, 150:250].mean() > 100


def test_crop_to_content_shrinks_blank_image(preprocessor):
    img = make_test_image()
    gray = preprocessor.to_grayscale(img)
    binary = preprocessor.binarize(gray)
    cropped = preprocessor.crop_to_content(binary, margin=5)
    assert cropped.shape[0] <= binary.shape[0]
    assert cropped.shape[1] <= binary.shape[1]


def test_full_pipeline_runs_end_to_end(preprocessor):
    img = make_test_image()
    result = preprocessor.process(image_bgr=img)
    assert result.cropped.ndim == 2
    assert result.cropped.size > 0
    assert isinstance(result.deskew_angle_deg, float)


def test_process_requires_path_or_array(preprocessor):
    with pytest.raises(ValueError):
        preprocessor.process()
