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
    deskew_skipped: bool = False   # True when the sample had too few characters to estimate skew (see process())
    n_chars_for_deskew: int = 0    # character count the skip decision was based on


class Preprocessor:
    """Modular preprocessing pipeline: load -> clean -> binarize -> deskew -> crop."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        self.cfg = cfg["preprocessing"]
        self._min_component_area = cfg.get("segmentation", {}).get("min_component_area", 20)

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
        """Estimate the rotation (degrees) that levels the text baseline; 0.0 if there is no clear winner.

        Row-projection-profile search: rotate the ink pixels by each candidate angle and keep the angle
        whose horizontal ink histogram is sharpest (sum of squared row counts), i.e. where text rows
        line up. The returned angle is the correction to hand straight to `deskew` (same
        `cv2.getRotationMatrix2D` sign convention), searched within +/- `deskew_angle_search_deg`
        at `deskew_angle_step_deg` resolution.

        This replaces a `cv2.minAreaRect` estimate over all ink, which fits a rectangle to the
        convex hull of the whole word and is therefore dominated by where the tallest ascender and
        deepest descender happen to sit. Measured on a perfectly horizontal 'quick' it returned
        -7.8 deg (and -11.3 deg on the committed sample, which has 3.5 deg of real tilt baked in),
        versus -0.6 deg on a full sentence: the error is a function of word shape, not of skew.
        Slant of individual letters does not affect row sums, so italic writing is not mistaken
        for tilt. The estimate is still weak for very short samples; `process` skips deskew entirely
        below `deskew_min_chars`.
        """
        ys, xs = np.nonzero(binary)
        if len(xs) < 10:
            return 0.0
        max_pts = 60000
        if len(xs) > max_pts:  # subsample for speed; deterministic stride, not random
            stride = len(xs) // max_pts + 1
            xs, ys = xs[::stride], ys[::stride]
        h, w = binary.shape[:2]
        x = xs.astype(np.float64) - w / 2.0
        y = ys.astype(np.float64) - h / 2.0

        limit = float(self.cfg["deskew_angle_search_deg"])
        step = float(self.cfg.get("deskew_angle_step_deg", 0.25))
        angles = np.arange(-limit, limit + step / 2, step)
        scores = np.empty(len(angles))
        for i, a in enumerate(angles):
            r = np.radians(a)
            # y-coordinate after cv2.getRotationMatrix2D(center, a, 1): y' = -sin(a)*x + cos(a)*y
            y_rot = -np.sin(r) * x + np.cos(r) * y
            counts = np.bincount(np.round(y_rot - y_rot.min()).astype(np.int64))
            scores[i] = float(np.sum(counts.astype(np.float64) ** 2))

        best = int(np.argmax(scores))
        zero_score = scores[int(np.argmin(np.abs(angles)))]
        if scores[best] < 1.01 * zero_score:  # no meaningful sharpening -> do not rotate on noise
            return 0.0
        return float(angles[best])

    def count_components(self, binary: np.ndarray) -> int:
        """Connected components above the noise floor. Only a stand-in for the character count when the
        caller does not know it (it over-counts dotted/multi-part letters)."""
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= self._min_component_area))

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

    def process(
        self, path: str | None = None, image_bgr: np.ndarray | None = None, n_chars: int | None = None
    ) -> PreprocessingResult:
        """Run the full preprocessing pipeline end to end.

        Provide exactly one of `path` or `image_bgr`.

        `n_chars` is the number of characters in the sample when known (e.g. the length of the typed
        label). Deskew is skipped entirely, angle 0.0, when it is below `deskew_min_chars`: a handful
        of letters does not determine a baseline angle, and rotating on a bad estimate does more harm
        than leaving a small tilt (it shears the dot of an 'i' off its stem). If `n_chars` is None the
        connected-component count is used instead.
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
        chars_for_deskew = n_chars if n_chars is not None else self.count_components(binary)
        skip_deskew = chars_for_deskew < self.cfg["deskew_min_chars"]
        angle = 0.0 if skip_deskew else self.estimate_skew_angle(binary)
        deskewed = self.deskew(binary, angle)
        cropped = self.crop_to_content(deskewed)
        if skip_deskew:
            logger.info("Deskew skipped: %d characters < deskew_min_chars=%d", chars_for_deskew, self.cfg["deskew_min_chars"])
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
            deskew_skipped=skip_deskew,
            n_chars_for_deskew=int(chars_for_deskew),
        )
