"""Glyph geometry: every glyph, observed or generated, carries `GlyphMetrics`.

Why: glyph bitmaps used to be plain arrays whose pixel size meant different
things depending on where they came from (a real crop ~100 px tall, a VAE
output 28x28 with EMNIST's fill-the-box normalisation, a baseline glyph forced
into one fixed box), and the renderer aligned bottom edges. `GlyphMetrics` says
where a bitmap sits relative to the writing line, so the renderer can put every
glyph on one baseline at one x-height regardless of its source resolution.

Conventions (image coordinates: x right, **y DOWN**; the baseline is y = 0):
  * `x_height`        height of the writer's x-height band, in THIS bitmap's pixels.
                      Scaling a bitmap by `target / x_height` puts it at the target x-height.
  * `baseline_offset` signed distance from the bitmap's bottom edge UP to the baseline, in the
                      bitmap's pixels: 0 for a letter standing on the line, **negative when the
                      ink drops below it** (g j p q y descenders).
  * `bbox`            (x, y, w, h) of the ink relative to the pen origin on the baseline:
                      x = left side bearing, y = top edge (negative = above the baseline),
                      w/h = bitmap size. Consistency: bbox.y + bbox.h == -baseline_offset.
  * `advance_width`   horizontal pen advance in the bitmap's pixels (ink width plus side bearings).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

X_HEIGHT_CHARS = frozenset("acemnorsuvwxz")   # ink stays within the x-height band
ASCENDER_CHARS = frozenset("bdfhkl")          # rise above it
DESCENDER_CHARS = frozenset("gpqy")           # drop below the baseline
# 't' (short ascender), 'i' (dot) and 'j' (dot + descender) are handled individually.


def glyph_class(char: str) -> str:
    """'x' | 'ascender' | 'descender' | 't' | 'i' | 'j' | 'other' for a lower-case letter."""
    c = char.lower()
    if c in X_HEIGHT_CHARS:
        return "x"
    if c in ASCENDER_CHARS:
        return "ascender"
    if c in DESCENDER_CHARS:
        return "descender"
    return c if c in ("t", "i", "j") else "other"


@dataclass(frozen=True)
class GlyphPriors:
    """Typographic PRIORS (not measurements), used only where a bitmap carries no absolute size:
    the VAE's 28x28 output, and line metrics for samples with no x-height-only letter to measure.
    Expressed in x-heights. Tune via the `glyph_metrics:` block of config.yaml."""

    ascender_ratio: float = 2.0         # total height of b d f h k l
    t_ratio: float = 1.5                # total height of t
    dotted_ratio: float = 1.8           # total height of i (stem + dot)
    descender_depth_ratio: float = 0.8  # depth below the baseline of g j p q y
    spacing_max_ratio: float = 1.0      # cap on the writer's inter-glyph gap

    @classmethod
    def from_config(cls, config: dict | None) -> "GlyphPriors":
        block = (config or {}).get("glyph_metrics", {}) or {}
        return cls(**{k: float(v) for k, v in block.items() if k in cls.__dataclass_fields__})

    def descender_depth(self, char: str) -> float:
        """Depth below the baseline, in x-heights (0 for letters that stand on the line)."""
        return self.descender_depth_ratio if glyph_class(char) in ("descender", "j") else 0.0

    def total_height_ratio(self, char: str) -> float:
        """Whole-glyph height (ink top to ink bottom) in x-heights."""
        cls_ = glyph_class(char)
        if cls_ == "ascender":
            return self.ascender_ratio
        if cls_ == "t":
            return self.t_ratio
        if cls_ == "i":
            return self.dotted_ratio
        if cls_ == "descender":
            return 1.0 + self.descender_depth_ratio
        if cls_ == "j":
            return self.dotted_ratio + self.descender_depth_ratio
        return 1.0


@dataclass(frozen=True)
class GlyphMetrics:
    advance_width: float
    x_height: float
    baseline_offset: float
    bbox: tuple[float, float, float, float]   # (x, y, w, h), see module docstring

    def __post_init__(self):
        if self.x_height <= 0:
            raise ValueError(f"x_height must be positive, got {self.x_height}")
        if self.advance_width <= 0:
            raise ValueError(f"advance_width must be positive, got {self.advance_width}")
        x, y, w, h = self.bbox
        if w < 0 or h < 0:
            raise ValueError(f"bbox must have non-negative size, got {self.bbox}")
        if abs((y + h) + self.baseline_offset) > 1e-3:
            raise ValueError(
                f"baseline_offset ({self.baseline_offset}) is inconsistent with bbox {self.bbox}: "
                "bbox.y + bbox.h must equal -baseline_offset"
            )

    def scaled(self, k: float) -> "GlyphMetrics":
        """Metrics of the same glyph after the bitmap is resized by factor k."""
        x, y, w, h = self.bbox
        return replace(
            self, advance_width=self.advance_width * k, x_height=self.x_height * k,
            baseline_offset=self.baseline_offset * k, bbox=(x * k, y * k, w * k, h * k),
        )

    def to_dict(self) -> dict:
        return {"advance_width": self.advance_width, "x_height": self.x_height,
                "baseline_offset": self.baseline_offset, "bbox": list(self.bbox)}

    @classmethod
    def from_dict(cls, d: dict) -> "GlyphMetrics":
        return cls(float(d["advance_width"]), float(d["x_height"]), float(d["baseline_offset"]),
                   tuple(float(v) for v in d["bbox"]))


@dataclass
class GlyphBitmap:
    """A glyph image (uint8, ink=255 on 0, tight to the ink) plus its metrics."""

    image: np.ndarray
    metrics: GlyphMetrics

    def __post_init__(self):
        if self.image.ndim != 2:
            raise ValueError(f"glyph image must be 2-D, got shape {self.image.shape}")
        h, w = self.image.shape
        if abs(self.metrics.bbox[2] - w) > 0.5 or abs(self.metrics.bbox[3] - h) > 0.5:
            raise ValueError(f"metrics.bbox {self.metrics.bbox} does not match image size {(w, h)}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape


def tight_crop(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop to the ink bounding box. Returns (crop, x0, y0); a blank image gives a 1x1 zero crop at (0, 0)."""
    coords = cv2.findNonZero((image > 0).astype(np.uint8))
    if coords is None:
        return np.zeros((1, 1), dtype=np.uint8), 0, 0
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w], x, y


def make_bitmap(image: np.ndarray, x_height: float, baseline_row: float, spacing: float) -> GlyphBitmap:
    """Wrap a tight-cropped `image`.

    Args:
        x_height: the writer's x-height in this image's pixels.
        baseline_row: y (rows from the TOP of `image`) of the writing baseline; > image height means the
            ink floats above the line, < image height means it drops below it.
        spacing: total side bearing to add to the ink width (half on each side), same pixels.
    """
    h, w = image.shape
    spacing = max(0.0, float(spacing))
    return GlyphBitmap(image, GlyphMetrics(
        advance_width=w + spacing,
        x_height=float(x_height),
        baseline_offset=float(baseline_row) - h,
        bbox=(spacing / 2.0, -float(baseline_row), float(w), float(h)),
    ))


def writer_spacing_ratio(mean_char_spacing_px: float, x_height_px: float, priors: GlyphPriors) -> float:
    """The writer's inter-glyph gap in x-heights, clamped to [0, priors.spacing_max_ratio]."""
    if x_height_px <= 0:
        return 0.0
    return float(np.clip(mean_char_spacing_px / x_height_px, 0.0, priors.spacing_max_ratio))


def bitmap_from_generated(char: str, image: np.ndarray, spacing_ratio: float, priors: GlyphPriors) -> GlyphBitmap:
    """Metrics for a VAE-generated glyph.

    EMNIST-style output fills its box regardless of letter, so the bitmap has NO absolute size
    information: an 'a' and a 'k' are both ~20 px tall. The size relationship is therefore taken from
    typographic priors (`GlyphPriors`): the x-height is the crop height divided by the letter's typical
    height in x-heights, and descenders hang below the baseline by the prior depth. This is a stated
    approximation, not something the model learned.
    """
    crop, _, _ = tight_crop(np.asarray(image))
    h = crop.shape[0]
    x_height = h / priors.total_height_ratio(char)
    baseline_row = h - priors.descender_depth(char) * x_height
    return make_bitmap(crop, x_height, baseline_row, spacing_ratio * x_height)
