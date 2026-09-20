"""Image preprocessing pipeline for handwritten input.

Converts a raw uploaded photo/scan of handwriting into a clean, deskewed,
binarized image with ink=255 (white) on a black (0) background — the
convention every downstream module (segmentation, feature extraction)
assumes.

The pipeline is intentionally split into small, independently testable
steps (`Preprocessor.<step>`) rather than one monolithic function, so any
step can be swapped out later (e.g. a learned binarization model) without
touching the rest of the system.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Container for every intermediate artifact of the pipeline.

    Keeping intermediates (not just the final binary image) makes the
    pipeline debuggable and lets the UI show a "what happened at each
    step" preview.
    """

    original_bgr: np.ndarray
    gray: np.ndarray
    denoised: np.ndarray
    contrast_normalized: np.ndarray
    binary: np.ndarray          # ink=255, background=0, pre-deskew
    deskew_angle_deg: float
    deskewed: np.ndarray        # ink=255, background=0, post-deskew
    cropped: np.ndarray         # deskewed image cropped to the ink bounding box


class Preprocessor:
    """Modular preprocessing pipeline: load -> clean -> binarize -> deskew -> crop."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        self.cfg = cfg["preprocessing"]

    # -- individual steps -------------------------------------------------

    def load_image(self, path: str) -> np.ndarray:
        """Load an image from disk as BGR. Raises if the file can't be read."""
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {path!r}")
        return self._resize_to_target_height(image)

    def load_image_from_array(self, image_bgr: np.ndarray) -> np.ndarray:
        """Accept an already-decoded BGR array (e.g. from a Streamlit upload)."""
        return self._resize_to_target_height(image_bgr)

    def _resize_to_target_height(self, image: np.ndarray) -> np.ndarray:
        target_h = self.cfg["target_line_height"]
        h, w = image.shape[:2]
        if h == 0:
            return image
        scale = target_h / h
        # Only upscale small phone-photo crops; don't blow up already-large scans excessively.
        scale = min(scale, 4.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def to_grayscale(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """Remove sensor/paper-texture noise while preserving stroke edges."""
        k = self.cfg["median_filter_kernel"]
        denoised = cv2.medianBlur(gray, k)
        denoised = cv2.fastNlMeansDenoising(denoised, h=7, templateWindowSize=7, searchWindowSize=21)
        return denoised

    def normalize_contrast(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE contrast normalization — evens out uneven lighting/shadows in phone photos."""
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive threshold + Otsu fallback. Returns ink=255, background=0."""
        block_size = self.cfg["adaptive_thresh_block_size"]
        if block_size % 2 == 0:
            block_size += 1
        C = self.cfg["adaptive_thresh_C"]

        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            block_size, C,
        )
        # Otsu as a global sanity check; combine via AND to suppress adaptive-threshold
        # speckle noise in blank paper regions while keeping true ink.
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined = cv2.bitwise_and(adaptive, otsu)

        # Morphological open to remove single-pixel speckle noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        return cleaned

    def estimate_skew_angle(self, binary: np.ndarray) -> float:
        """Estimate baseline skew via minAreaRect over all ink pixels.

        Searches within +/- `deskew_angle_search_deg` to avoid the classic
        minAreaRect failure mode of returning a near-90-degree angle for
        text that is only slightly tilted.
        """
        coords = cv2.findNonZero(binary)
        if coords is None or len(coords) < 10:
            return 0.0
        angle = cv2.minAreaRect(coords)[-1]
        # cv2.minAreaRect angle convention: normalize into [-45, 45].
        if angle < -45:
            angle = 90 + angle
        max_search = self.cfg["deskew_angle_search_deg"]
        if abs(angle) > max_search:
            return 0.0
        return float(angle)

    def deskew(self, binary: np.ndarray, angle_deg: float) -> np.ndarray:
        if abs(angle_deg) < 0.1:
            return binary
        h, w = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            binary, M, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def crop_to_content(self, binary: np.ndarray, margin: int = 10) -> np.ndarray:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return binary
        x, y, w, h = cv2.boundingRect(coords)
        H, W = binary.shape[:2]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        return binary[y0:y1, x0:x1]

    # -- full pipeline ------------------------------------------------------

    def process(self, path: str | None = None, image_bgr: np.ndarray | None = None) -> PreprocessingResult:
        """Run the full preprocessing pipeline end to end.

        Provide exactly one of `path` or `image_bgr`.
        """
        if image_bgr is not None:
            original = self._resize_to_target_height(image_bgr)
        elif path is not None:
            original = self.load_image(path)
        else:
            raise ValueError("Provide either `path` or `image_bgr`.")

        logger.info("Preprocessing image of shape %s", original.shape)
        gray = self.to_grayscale(original)
        denoised = self.denoise(gray)
        contrast = self.normalize_contrast(denoised)
        binary = self.binarize(contrast)
        angle = self.estimate_skew_angle(binary)
        deskewed = self.deskew(binary, angle)
        cropped = self.crop_to_content(deskewed)
        logger.info("Deskew angle: %.2f deg, final size: %s", angle, cropped.shape)

        return PreprocessingResult(
            original_bgr=original,
            gray=gray,
            denoised=denoised,
            contrast_normalized=contrast,
            binary=binary,
            deskew_angle_deg=angle,
            deskewed=deskewed,
            cropped=cropped,
        )
