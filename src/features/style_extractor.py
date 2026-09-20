"""Stage-1 rule-based handwriting feature extraction (non-neural baseline).

Computes explicit, human-interpretable measurements from segmented glyphs:
height, width, slant, stroke width, curvature, spacing, baseline. This is
the "Simple glyph extraction + nearest style transformation" BASELINE
referenced in docs/experiments.md, and it also feeds the UI's "Handwriting
style analysis" panel. It is deliberately independent of the neural style
encoder (src/style_encoder/) so the two can be compared quantitatively.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.features.line_metrics import LineMetrics, estimate_line_metrics
from src.generator.glyph import GlyphPriors
from src.segmentation.segmenter import Glyph


@dataclass
class GlyphFeatures:
    char: str
    height_px: int
    width_px: int
    aspect_ratio: float
    slant_deg: float
    stroke_width_px: float
    ink_density: float          # fraction of bbox filled with ink
    curvature_score: float      # 0 (very angular) .. 1 (very round)


@dataclass
class StyleProfile:
    """Aggregated, writer-level handwriting style summary."""

    glyph_features: list[GlyphFeatures]
    mean_height_px: float
    mean_width_px: float
    mean_aspect_ratio: float
    mean_slant_deg: float
    slant_std_deg: float
    mean_stroke_width_px: float
    mean_curvature_score: float
    mean_char_spacing_px: float
    baseline_y: int
    x_height_top_y: int
    feature_vector: np.ndarray = field(repr=False)  # compact numeric summary for downstream use
    x_height_px: float = 0.0                        # writer x-height (source pixels); 0.0 = not estimated
    line_metrics: LineMetrics | None = None         # the float-precision estimate behind baseline_y/x_height_*


class StyleFeatureExtractor:
    """Extracts per-glyph and writer-level handwriting style features."""

    def __init__(self, priors: GlyphPriors | None = None):
        self.priors = priors or GlyphPriors()

    def _stroke_width(self, glyph_bin: np.ndarray) -> float:
        """Median stroke width via distance transform (2x the medial-axis radius)."""
        if glyph_bin.sum() == 0:
            return 0.0
        # Pad with background: a tight crop that is entirely ink (a solid stem) has no zero
        # pixel for the transform to measure against and returns ~1e37, which overflows.
        padded = cv2.copyMakeBorder(glyph_bin, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        # Skeleton-ish: only consider local maxima along the ink to avoid
        # averaging in the tapered edges of strokes.
        ink_distances = dist[padded > 0]
        if ink_distances.size == 0:
            return 0.0
        return float(np.percentile(ink_distances, 75) * 2)

    def _slant_deg(self, glyph_bin: np.ndarray) -> float:
        """Estimate stroke slant from the dominant orientation of ink pixels via PCA."""
        ys, xs = np.nonzero(glyph_bin)
        if len(xs) < 5:
            return 0.0
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        dx, dy = principal
        if dy == 0:
            return 0.0
        # Angle from vertical (0 deg = perfectly upright stroke), image y grows downward.
        angle = np.degrees(np.arctan2(dx, -dy))
        # Fold to [-90, 90], report deviation from vertical.
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return float(angle)

    def _curvature_score(self, glyph_bin: np.ndarray) -> float:
        """Roundness proxy: contour perimeter^2 / (4*pi*area) inverted into [0,1].

        A perfect circle has isoperimetric ratio 1.0; angular/spiky shapes have
        larger ratios. We map this to a bounded 0..1 "curvature score" where
        1 = very round, 0 = very angular.
        """
        contours, _ = cv2.findContours(glyph_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            return 0.0
        ratio = (perimeter ** 2) / (4 * np.pi * area)
        return float(1.0 / ratio) if ratio >= 1 else 1.0

    def extract_glyph_features(self, char: str, glyph_bin: np.ndarray) -> GlyphFeatures:
        h, w = glyph_bin.shape[:2]
        ink_density = float((glyph_bin > 0).sum() / max(1, h * w))
        return GlyphFeatures(
            char=char,
            height_px=h,
            width_px=w,
            aspect_ratio=w / h if h > 0 else 0.0,
            slant_deg=self._slant_deg(glyph_bin),
            stroke_width_px=self._stroke_width(glyph_bin),
            ink_density=ink_density,
            curvature_score=self._curvature_score(glyph_bin),
        )

    def extract_style_profile_excluding(
        self, chars: list[str], glyphs: list[Glyph], exclude_char: str
    ) -> StyleProfile:
        """Style profile rebuilt from every glyph EXCEPT those labelled `exclude_char`.

        Used for leave-one-out scoring. Nothing derived from an excluded glyph may
        influence the result, which means more than dropping it from the feature list:
          * the baseline and x-height are re-estimated from the retained glyphs only
            (`extract_style_profile` does this from the glyphs it is given);
          * spacing is measured only between glyphs that were adjacent in the original
            line AND are both retained (the gap that straddles a removed glyph is
            contaminated by that glyph's width, so it is skipped);
          * every occurrence is removed (e.g. both 'l' in "hello").
        """
        keep = [i for i, c in enumerate(chars) if c.lower() != exclude_char.lower()]
        if not keep:
            raise ValueError(f"Excluding {exclude_char!r} leaves no glyphs to build a style profile from")
        kept = set(keep)
        spacings = [
            glyphs[i + 1].bbox[0] - (glyphs[i].bbox[0] + glyphs[i].bbox[2])
            for i in range(len(glyphs) - 1) if i in kept and (i + 1) in kept
        ]
        return self.extract_style_profile(
            [chars[i] for i in keep], [glyphs[i] for i in keep], 0, 0, spacings=spacings
        )

    def extract_style_profile(
        self, chars: list[str], glyphs: list[Glyph], baseline_y: int, x_height_top_y: int,
        spacings: list[float] | None = None,
    ) -> StyleProfile:
        """Aggregate per-glyph features into a writer profile.

        `baseline_y` / `x_height_top_y` are the segmenter's medians over all glyph boxes and are only
        kept for callers that have nothing better; the profile's own `baseline_y`, `x_height_top_y` and
        `x_height_px` come from `estimate_line_metrics` over the labelled glyphs given here (letters
        that live in the x-height band give the exact values; see src/features/line_metrics.py).
        """
        glyph_features = [
            self.extract_glyph_features(c, g.image) for c, g in zip(chars, glyphs)
        ]

        if spacings is None:  # default: gaps between consecutive glyphs in the given list
            spacings = []
            for i in range(len(glyphs) - 1):
                gap = glyphs[i + 1].bbox[0] - (glyphs[i].bbox[0] + glyphs[i].bbox[2])
                spacings.append(gap)

        heights = [f.height_px for f in glyph_features]
        widths = [f.width_px for f in glyph_features]
        aspects = [f.aspect_ratio for f in glyph_features]
        slants = [f.slant_deg for f in glyph_features]
        strokes = [f.stroke_width_px for f in glyph_features]
        curvatures = [f.curvature_score for f in glyph_features]

        mean_height = float(np.mean(heights)) if heights else 0.0
        mean_width = float(np.mean(widths)) if widths else 0.0
        mean_aspect = float(np.mean(aspects)) if aspects else 0.0
        mean_slant = float(np.mean(slants)) if slants else 0.0
        std_slant = float(np.std(slants)) if slants else 0.0
        mean_stroke = float(np.mean(strokes)) if strokes else 0.0
        mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0
        mean_spacing = float(np.mean(spacings)) if spacings else 0.0

        feature_vector = np.array(
            [mean_height, mean_width, mean_aspect, mean_slant, std_slant,
             mean_stroke, mean_curvature, mean_spacing],
            dtype=np.float32,
        )

        line = estimate_line_metrics(chars, glyphs, self.priors) if glyphs else None
        if line is not None:
            baseline_y = int(round(line.baseline_y))
            x_height_top_y = int(round(line.baseline_y - line.x_height))

        return StyleProfile(
            glyph_features=glyph_features,
            mean_height_px=mean_height,
            mean_width_px=mean_width,
            mean_aspect_ratio=mean_aspect,
            mean_slant_deg=mean_slant,
            slant_std_deg=std_slant,
            mean_stroke_width_px=mean_stroke,
            mean_curvature_score=mean_curvature,
            mean_char_spacing_px=mean_spacing,
            baseline_y=baseline_y,
            x_height_top_y=x_height_top_y,
            feature_vector=feature_vector,
            x_height_px=line.x_height if line is not None else 0.0,
            line_metrics=line,
        )
