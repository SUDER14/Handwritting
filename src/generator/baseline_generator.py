"""BASELINE glyph generator: "simple glyph extraction + nearest style transformation".

This is the non-generative baseline described in docs/experiments.md,
implemented per the MVP fallback in the design brief (section 25):

- Observed characters (present in the user's sample) are used AS-IS —
  their real handwritten pixels, just size-normalized.
- Unobserved characters are synthesized by taking a neutral reference font
  glyph and warping it (scale, shear/slant, stroke-width morphology) to
  match the writer's measured StyleProfile.

IMPORTANT — this is intentionally NOT presented as the final AI system.
It contains no learned representation of handwriting style; it is a
deterministic image-processing transform, used as (a) a fast fallback that
always produces a full alphabet even before any model is trained, and
(b) the baseline that the neural style-encoder + conditional-generator
pipeline (src/style_encoder/, src/generator/conditional_generator.py) is
compared against.
"""
from __future__ import annotations

import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.features.style_extractor import StyleProfile

REFERENCE_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoesc.ttf",
    "C:/Windows/Fonts/comic.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
RENDER_SIZE = 200


def _load_reference_font(size: int = RENDER_SIZE) -> ImageFont.FreeTypeFont:
    for path in REFERENCE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_reference_glyph(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render a character from the neutral reference font, tightly cropped, ink=255."""
    canvas = RENDER_SIZE * 2
    img = Image.new("L", (canvas, canvas), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - w) / 2 - bbox[0]
    y = (canvas - h) / 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    arr = np.array(img)
    coords = cv2.findNonZero(arr)
    if coords is None:
        return np.zeros((10, 10), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    return arr[y:y + h, x:x + w]


def _measure_stroke_width(glyph_bin: np.ndarray) -> float:
    if glyph_bin.sum() == 0:
        return 1.0
    padded = cv2.copyMakeBorder(glyph_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # see style_extractor._stroke_width
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    ink_distances = dist[padded > 0]
    return float(np.percentile(ink_distances, 75) * 2) if ink_distances.size else 1.0


def _adjust_stroke_width(glyph_bin: np.ndarray, target_width_px: float) -> np.ndarray:
    """Dilate or erode the glyph so its stroke width matches the target (in the glyph's own pixel scale)."""
    current = _measure_stroke_width(glyph_bin)
    if current <= 0:
        return glyph_bin
    ratio = target_width_px / current
    iterations = int(round(abs(ratio - 1) * 3))
    if iterations == 0:
        return glyph_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if ratio > 1:
        return cv2.dilate(glyph_bin, kernel, iterations=iterations)
    return cv2.erode(glyph_bin, kernel, iterations=iterations)


def _apply_shear(glyph_bin: np.ndarray, slant_deg: float) -> np.ndarray:
    """Shear the glyph horizontally to emulate a slant. slant_deg: 0 = upright."""
    if abs(slant_deg) < 0.5:
        return glyph_bin
    h, w = glyph_bin.shape[:2]
    shear = np.tan(np.radians(slant_deg))
    pad = int(abs(shear) * h) + 2
    M = np.array([[1, shear, -min(0, shear) * h + 0], [0, 1, 0]], dtype=np.float32)
    padded = cv2.copyMakeBorder(glyph_bin, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    sheared = cv2.warpAffine(
        padded, M, (padded.shape[1], padded.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    coords = cv2.findNonZero(sheared)
    if coords is None:
        return glyph_bin
    x, y, w2, h2 = cv2.boundingRect(coords)
    return sheared[y:y + h2, x:x + w2]


def _resize_to_target(glyph_bin: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    target_h, target_w = max(4, target_h), max(4, target_w)
    return cv2.resize(glyph_bin, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _jitter(glyph_bin: np.ndarray, rotation_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Small random rotation to give repeated generations a natural, non-identical look."""
    angle = rng.uniform(-rotation_deg, rotation_deg)
    h, w = glyph_bin.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(glyph_bin, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class BaselineGlyphGenerator:
    """Generates the missing alphabet via style-transformed reference-font glyphs.

    Observed glyphs are preserved as real handwriting pixels (normalized).
    """

    def __init__(self, config: dict | None = None):
        self.font = _load_reference_font()
        self.rng = np.random.default_rng(42)
        self.variants_per_char = (config or {}).get("generation", {}).get("variants_per_char", 3)
        self.jitter_rotation_deg = (config or {}).get("generation", {}).get("jitter_rotation_deg", 1.5)

    def generate_observed(self, glyph_bin: np.ndarray, style: StyleProfile) -> np.ndarray:
        """Normalize a real observed glyph to the writer's mean size (no shape change)."""
        target_h = int(round(style.mean_height_px))
        aspect = glyph_bin.shape[1] / max(1, glyph_bin.shape[0])
        target_w = max(4, int(round(target_h * aspect)))
        return _resize_to_target(glyph_bin, target_h, target_w)

    def generate_unobserved(self, char: str, style: StyleProfile) -> list[np.ndarray]:
        """Synthesize `variants_per_char` glyph variants for a character not in the sample."""
        base = _render_reference_glyph(char, self.font)
        target_h = int(round(style.mean_height_px))
        target_w = max(4, int(round(target_h * style.mean_aspect_ratio)))

        variants = []
        for _ in range(self.variants_per_char):
            g = _resize_to_target(base, target_h, target_w)
            g = _adjust_stroke_width(g, style.mean_stroke_width_px * (target_h / max(1, style.mean_height_px)))
            g = _apply_shear(g, style.mean_slant_deg)
            g = _resize_to_target(g, target_h, target_w)
            g = _jitter(g, self.jitter_rotation_deg, self.rng)
            variants.append(g)
        return variants

    def generate_alphabet(
        self,
        observed_chars: dict[str, np.ndarray],
        style: StyleProfile,
        charset: str = string.ascii_lowercase,
    ) -> dict[str, list[np.ndarray]]:
        """Build the full alphabet: real glyphs for observed chars, synthesized for the rest."""
        alphabet: dict[str, list[np.ndarray]] = {}
        for char in charset:
            if char in observed_chars:
                alphabet[char] = [self.generate_observed(observed_chars[char], style)]
            else:
                alphabet[char] = self.generate_unobserved(char, style)
        return alphabet
