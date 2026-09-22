"""BASELINE glyph generator: "simple glyph extraction + nearest style transformation".

This is the non-generative baseline described in docs/experiments.md,
implemented per the MVP fallback in the design brief (section 25):

- Observed characters (present in the user's sample) are used AS-IS: their
  real handwritten pixels at native resolution, with the metrics measured
  from the sample (see src/features/line_metrics.py).
- Unobserved characters are synthesized from a neutral reference font glyph,
  transformed to match the writer's StyleProfile: horizontal stretch (letter
  width relative to x-height), stroke width, and slant.

Geometry: nothing is forced into a common box any more. Each glyph keeps its
own natural proportions (a narrow 'i', a wide 'm', an ascender 'k', a
descender 'g'), and carries `GlyphMetrics` saying where the writing baseline
falls in its bitmap (from the font's own baseline, tracked through every
transform). The renderer normalises to a common x-height and places glyphs by
baseline (src/renderer/text_renderer.py).

IMPORTANT: this is intentionally NOT presented as the final AI system. It
contains no learned representation of handwriting style; it is a deterministic
image-processing transform, used as (a) a fast fallback that always produces a
full alphabet even before any model is trained, and (b) the baseline that the
neural style-encoder pipeline (src/style_encoder/) is compared against.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.features.style_extractor import StyleProfile
from src.generator.glyph import GlyphBitmap, GlyphPriors, make_bitmap, tight_crop, writer_spacing_ratio

REFERENCE_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoesc.ttf",
    "C:/Windows/Fonts/comic.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
RENDER_SIZE = 200
MAX_STROKE_ITERATIONS = 6
WIDTH_SCALE_RANGE = (0.5, 1.8)


_FONT_SLANT_CACHE: dict[str, float] = {}


def font_slant_deg(font: ImageFont.FreeTypeFont, cache_dir: str | None = "data/processed") -> float:
    """Slant of the reference font's own letters, measured ONCE with our own slant estimator
    (StyleFeatureExtractor._slant_deg, the same statistic the writer's slant is measured with: the mean over a-z)
    and cached in memory and in <cache_dir>/font_slant.json. The baseline shears by (writer slant - font slant): the
    font is itself slanted (Segoe Script ~ +20 deg), so applying the writer's full slant on top of it double-counts."""
    import json
    from pathlib import Path

    from src.features.style_extractor import StyleFeatureExtractor
    key = f"{getattr(font, 'path', 'default')}@{RENDER_SIZE}"
    if key in _FONT_SLANT_CACHE:
        return _FONT_SLANT_CACHE[key]
    disk = None
    if cache_dir:
        from src.utils.config import resolve_path
        disk = resolve_path(cache_dir) / "font_slant.json"
        try:
            _FONT_SLANT_CACHE.update(json.loads(disk.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        if key in _FONT_SLANT_CACHE:
            return _FONT_SLANT_CACHE[key]
    ex = StyleFeatureExtractor()
    value = float(np.mean([ex._slant_deg(_render_font_glyph(c, font)[0]) for c in string.ascii_lowercase]))
    _FONT_SLANT_CACHE[key] = value
    if disk is not None:
        try:
            disk.parent.mkdir(parents=True, exist_ok=True)
            disk.write_text(json.dumps(_FONT_SLANT_CACHE, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    return value


def _load_reference_font(size: int = RENDER_SIZE) -> ImageFont.FreeTypeFont:
    for path in REFERENCE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class _FontGlyph:
    image: np.ndarray      # tight crop, ink=255
    baseline_row: float    # rows from the top of `image` to the font's baseline
    x_height: float        # the font's x-height, same pixels


def _render_font_glyph(char: str, font: ImageFont.FreeTypeFont) -> tuple[np.ndarray, int]:
    """Render `char` anchored on its baseline; returns (tight crop, baseline row within the crop)."""
    canvas, baseline_y = RENDER_SIZE * 3, RENDER_SIZE * 2
    img = Image.new("L", (canvas, canvas), color=0)
    ImageDraw.Draw(img).text((RENDER_SIZE, baseline_y), char, fill=255, font=font, anchor="ls")
    crop, _, y0 = tight_crop(np.array(img))
    return crop, baseline_y - y0


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
    iterations = min(MAX_STROKE_ITERATIONS, int(round(abs(ratio - 1) * 3)))
    if iterations == 0:
        return glyph_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if ratio > 1:
        return cv2.dilate(glyph_bin, kernel, iterations=iterations)
    return cv2.erode(glyph_bin, kernel, iterations=iterations)


def _apply_shear(canvas: np.ndarray, slant_deg: float) -> np.ndarray:
    """Shear a padded glyph canvas horizontally by `slant_deg`, in the SAME sign convention as the slant estimator
    (StyleFeatureExtractor._slant_deg: positive = strokes lean right, i.e. the top of a stroke is right of its bottom).
    0 = unchanged. Shearing a vertical bar by +20 makes the estimator read +20 (tests/test_generator.py).

    x' = x - tan(slant) * y  (image y grows downward, so the top rows move right for a positive slant). Applied in
    place on a canvas that already has horizontal room, so the vertical layout -- and the baseline row -- is untouched.
    """
    if abs(slant_deg) < 0.5:
        return canvas
    h, w = canvas.shape[:2]
    shear = -np.tan(np.radians(slant_deg))
    pad = int(abs(shear) * h) + 2
    padded = cv2.copyMakeBorder(canvas, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    m = np.array([[1, shear, -min(0, shear) * h], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(padded, m, (padded.shape[1], padded.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _jitter(canvas: np.ndarray, baseline_row: float, rotation_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Small random rotation about the point (canvas centre x, baseline), so the baseline pixel stays put."""
    angle = rng.uniform(-rotation_deg, rotation_deg)
    h, w = canvas.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, float(baseline_row)), angle, 1.0)
    return cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class BaselineGlyphGenerator:
    """Generates the missing alphabet via style-transformed reference-font glyphs.

    Observed glyphs are preserved as real handwriting pixels at native resolution.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.font = _load_reference_font()
        self.rng = np.random.default_rng(42)
        self.variants_per_char = cfg.get("generation", {}).get("variants_per_char", 3)
        self.jitter_rotation_deg = cfg.get("generation", {}).get("jitter_rotation_deg", 1.5)
        self.priors = GlyphPriors.from_config(cfg)
        self._reference_cache: dict[str, _FontGlyph] = {}
        self._font_x_height: float | None = None
        self._font_slant_deg: float | None = None

    # -- reference font ------------------------------------------------------

    def _x_height(self) -> float:
        """The font's x-height = how far the top of 'x' rises ABOVE the baseline. (Not the crop height:
        the reference font is a cursive script whose 'x' has a tail below the line, which would inflate it.)"""
        if self._font_x_height is None:
            self._font_x_height = float(_render_font_glyph("x", self.font)[1])
        return self._font_x_height

    def _font_slant(self) -> float:
        """The reference font's own slant (measured once, cached); the writer's slant is applied relative to it."""
        if self._font_slant_deg is None:
            self._font_slant_deg = font_slant_deg(self.font)
        return self._font_slant_deg

    def _reference(self, char: str) -> _FontGlyph:
        if char not in self._reference_cache:
            image, baseline_row = _render_font_glyph(char, self.font)
            self._reference_cache[char] = _FontGlyph(image, float(baseline_row), self._x_height())
        return self._reference_cache[char]

    def _width_scale(self, style: StyleProfile) -> float:
        """Horizontal stretch so that letter width RELATIVE TO X-HEIGHT matches the writer's.

        Median over the writer's observed letters of
        (writer width / writer x-height) / (font width / font x-height); 1.0 with no observed letters.
        Replaces the old single mean aspect ratio that forced every letter into one box.
        """
        ratios = []
        for f in style.glyph_features:
            c = f.char.lower()
            if c in string.ascii_lowercase and f.width_px > 0:
                ref = self._reference(c)
                ratios.append((f.width_px / style.x_height_px) / (ref.image.shape[1] / ref.x_height))
        return float(np.clip(np.median(ratios), *WIDTH_SCALE_RANGE)) if ratios else 1.0

    # -- generation ------------------------------------------------------------

    def generate_unobserved(self, char: str, style: StyleProfile) -> list[GlyphBitmap]:
        """Synthesize `variants_per_char` glyph variants for a character not in the sample."""
        if style.x_height_px <= 0:
            raise ValueError("StyleProfile has no x_height_px; build it with StyleFeatureExtractor.extract_style_profile")
        ref = self._reference(char)
        scale_to_font = ref.x_height / style.x_height_px          # writer pixels -> font pixels
        width_scale = self._width_scale(style)
        target_stroke = style.mean_stroke_width_px * scale_to_font
        ratio = writer_spacing_ratio(style.mean_char_spacing_px, style.x_height_px, self.priors)

        variants = []
        for _ in range(self.variants_per_char):
            pad = max(8, ref.image.shape[0] // 6)
            canvas = cv2.copyMakeBorder(ref.image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            baseline_row = ref.baseline_row + pad                  # baseline row within the canvas (vertical ops keep it)

            if abs(width_scale - 1.0) > 1e-3:                      # horizontal-only resize: rows, so baseline, unchanged
                canvas = cv2.resize(canvas, (max(4, int(round(canvas.shape[1] * width_scale))), canvas.shape[0]),
                                    interpolation=cv2.INTER_AREA if width_scale < 1 else cv2.INTER_CUBIC)
                canvas = (canvas > 127).astype(np.uint8) * 255
            canvas = _adjust_stroke_width(canvas, target_stroke)
            canvas = _apply_shear(canvas, style.mean_slant_deg - self._font_slant())
            canvas = _jitter(canvas, baseline_row, self.jitter_rotation_deg, self.rng)

            crop, _, y0 = tight_crop(canvas)
            variants.append(make_bitmap(crop, ref.x_height, baseline_row - y0, ratio * ref.x_height))
        return variants

    def generate_alphabet(
        self,
        observed: dict[str, GlyphBitmap],
        style: StyleProfile,
        charset: str = string.ascii_lowercase,
    ) -> dict[str, list[GlyphBitmap]]:
        """Build the full alphabet: real glyphs (native scale, measured metrics) for observed chars,
        synthesized for the rest. Sizes are reconciled at render time by x-height, not here."""
        alphabet: dict[str, list[GlyphBitmap]] = {}
        for char in charset:
            alphabet[char] = [observed[char]] if char in observed else self.generate_unobserved(char, style)
        return alphabet
