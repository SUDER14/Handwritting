"""Per-writer handwriting descriptors for the Task 5.2 analysis, measured on a writer's REAL lines (native IAM scans).

  slant_deg     lean of the writing, degrees; POSITIVE = strokes lean to the right (top displaced right). Median over
                lines of the magnitude-weighted median orientation of ink edges within +-60 deg of vertical (edge
                tangent from a Sobel gradient of the blurred mask). Cross-check `lean_proj_deg`: the shear that sharpens
                the vertical ink projection (`segmentation.words.slant_angle`, search limit +-45 deg, saturates on some
                lines). Both are checked on a synthetic bar in the tests.
  stroke_px     stroke width in native pixels: 2 x the 75th percentile of the distance transform over ink pixels
                (the same definition as `features.style_extractor._stroke_width`), median over lines.
  cursiveness   mean connected components per word: 8-connected ink components of area >= `MIN_AREA` px per line,
                divided by the transcription's word count, averaged over lines. Fewer components per word = more joined
                (more cursive) writing. Dots of i/j count as components (same for every writer with the same text mix).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.words import slant_angle

MIN_AREA = 15


def binarize_ink(gray: np.ndarray) -> np.ndarray:
    """IAM scan (dark ink on light paper) -> uint8 ink mask, ink = 255."""
    _, ink = cv2.threshold(np.asarray(gray, np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def lean_proj_deg(ink: np.ndarray) -> float:
    return -slant_angle(ink)


def lean_deg(ink: np.ndarray) -> float:
    f = cv2.GaussianBlur((np.asarray(ink) > 127).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    m = mag > 0.2 * mag.max() if mag.max() > 0 else mag > 1
    if not m.any():
        return 0.0
    tx, ty = -gy[m], gx[m]                              # edge tangent = gradient rotated by 90 deg; orient it upward
    up = np.where(ty > 0, -1.0, 1.0)
    lean = np.degrees(np.arctan2(tx * up, -(ty * up)))
    sel = np.abs(lean) < 60
    if not sel.any():
        return 0.0
    order = np.argsort(lean[sel])
    w = np.cumsum(mag[m][sel][order])
    return float(lean[sel][order][np.searchsorted(w, w[-1] / 2)])


def stroke_width_px(ink: np.ndarray) -> float:
    b = (np.asarray(ink) > 127).astype(np.uint8)
    if not b.any():
        return 0.0
    padded = cv2.copyMakeBorder(b, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    return float(np.percentile(dist[padded > 0], 75) * 2)


def components_per_word(ink: np.ndarray, n_words: int) -> float | None:
    if n_words < 1:
        return None
    n, _, stats, _ = cv2.connectedComponentsWithStats((np.asarray(ink) > 127).astype(np.uint8), connectivity=8)
    comps = int((stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA).sum())
    return comps / n_words


def line_descriptors(gray: np.ndarray, transcription: str) -> dict:
    ink = binarize_ink(gray)
    return {"slant_deg": lean_deg(ink), "slant_proj_deg": lean_proj_deg(ink), "stroke_px": stroke_width_px(ink),
            "cursiveness": components_per_word(ink, len(transcription.split()))}


def writer_descriptors(lines: list[tuple[np.ndarray, str]]) -> dict:
    """lines = [(native gray, transcription)] of ONE writer -> medians / mean over lines."""
    rows = [line_descriptors(g, t) for g, t in lines]
    cc = [r["cursiveness"] for r in rows if r["cursiveness"] is not None]
    return {"n_lines": len(rows), "slant_deg": float(np.median([r["slant_deg"] for r in rows])),
            "slant_proj_deg": float(np.median([r["slant_proj_deg"] for r in rows])),
            "stroke_px": float(np.median([r["stroke_px"] for r in rows])),
            "cursiveness": float(np.mean(cc)) if cc else float("nan")}
