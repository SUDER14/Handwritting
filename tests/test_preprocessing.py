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


# -- deskew: estimator accuracy and the short-sample gate ---------------------------------------
import cv2  # noqa: E402

from src.utils.config import resolve_path  # noqa: E402


def _synthetic_text(text: str, rotate_deg: float = 0.0, scale: float = 2.6) -> np.ndarray:
    """Black text on white via OpenCV's built-in Hershey font (no system fonts needed), optionally
    rotated counter-clockwise by `rotate_deg` (cv2.getRotationMatrix2D convention)."""
    img = np.full((300, 1500, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    if rotate_deg:
        m = cv2.getRotationMatrix2D((750, 150), rotate_deg, 1.0)
        img = cv2.warpAffine(img, m, (1500, 300), borderValue=(255, 255, 255))
    return img


SENTENCE = "the quick brown fox jumps"   # 21 letters, well above deskew_min_chars


def test_skew_is_near_zero_on_horizontal_synthetic_text(preprocessor):
    result = preprocessor.process(image_bgr=_synthetic_text(SENTENCE), n_chars=21)
    assert not result.deskew_skipped
    assert abs(result.deskew_angle_deg) < 0.5


def test_estimator_alone_is_near_zero_on_a_horizontal_short_word(preprocessor):
    """Regression for the reported bug: cv2.minAreaRect returned -7.8 deg on horizontal 'quick' (word
    shape, not skew). The estimator itself must be ~0 there, independent of the short-sample gate."""
    img = preprocessor._resize_to_target_height(_synthetic_text("quick"))
    binary = preprocessor.binarize(preprocessor.normalize_contrast(preprocessor.denoise(
        preprocessor.to_grayscale(img))))
    assert abs(preprocessor.estimate_skew_angle(binary)) < 0.5


@pytest.mark.parametrize("tilt", [4.0, -6.0])
def test_estimator_recovers_a_known_tilt_with_the_right_sign(preprocessor, tilt):
    result = preprocessor.process(image_bgr=_synthetic_text(SENTENCE, rotate_deg=tilt), n_chars=21)
    assert result.deskew_angle_deg == pytest.approx(-tilt, abs=0.75)       # correction undoes the tilt
    assert abs(preprocessor.estimate_skew_angle(result.deskewed)) < 0.75    # and the output is level


def test_deskew_is_skipped_for_short_samples_even_when_tilted(preprocessor):
    n_min = preprocessor.cfg["deskew_min_chars"]
    tilted_word = _synthetic_text("quick", rotate_deg=5.0)
    result = preprocessor.process(image_bgr=tilted_word, n_chars=5)
    assert 5 < n_min and result.deskew_skipped and result.deskew_angle_deg == 0.0
    assert result.n_chars_for_deskew == 5
    # Same decision from the component-count fallback when the caller does not know the length.
    fallback = preprocessor.process(image_bgr=tilted_word)
    assert fallback.deskew_skipped and fallback.deskew_angle_deg == 0.0
    # The threshold is what decides: the same image with a large claimed length is deskewed.
    assert not preprocessor.process(image_bgr=tilted_word, n_chars=n_min).deskew_skipped


def test_committed_quick_sample_is_not_rotated_by_preprocessing(preprocessor):
    """data/samples/quick_sample.png used to come back as -11.35 deg. It has only 5 letters, so the
    gate leaves it alone."""
    img = cv2.imread(str(resolve_path("data/samples/quick_sample.png")), cv2.IMREAD_COLOR)
    result = preprocessor.process(image_bgr=img, n_chars=5)
    assert result.deskew_skipped and result.deskew_angle_deg == 0.0
