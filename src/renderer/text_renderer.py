"""Text-to-handwriting rendering engine (section 14 of the design brief).

Composes arbitrary text into an image using a personalized alphabet,
with natural (non-identical) variation: per-occurrence glyph-variant
selection, small baseline jitter, and slight rotation jitter — real
handwriting is never pixel-identical letter to letter.

GEOMETRY. Glyphs come from different sources at different pixel scales (real
crops ~100 px tall, VAE output 28 px, font-derived baseline glyphs), so pixel
size is meaningless. Each glyph carries `GlyphMetrics` (src/generator/glyph.py)
and the renderer works from those:

  * every glyph is scaled so its x-height equals `renderer.target_x_height_px`
    (a common x-height, NOT a common bounding box: an 'i' stays narrow, a 'k'
    stays tall, a 'g' keeps its tail);
  * glyphs are placed by BASELINE, not bottom edge: the ink's bottom sits at
    `-baseline_offset` below the baseline, so g j p q y drop below the line;
  * the pen advances by each glyph's `advance_width` (scaled) plus
    `renderer.char_spacing_px` of extra tracking.
Ascent, descent and baseline row are computed once over the whole alphabet, so
every rendered line has the same height and baseline and stacked lines align.

CURRENT SCOPE (documented limitation, see docs/limitations.md): the
active alphabet is lowercase a-z (per project priority). Uppercase
letters fall back to an upscaled lowercase glyph if no uppercase
alphabet has been generated; punctuation/digits are rendered as
whitespace-width gaps for now — both are architected as drop-in
extensions (section 31) once uppercase/digit alphabets are generated
the same way as lowercase.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.generator.glyph import GlyphBitmap
from src.utils.config import load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

UPPERCASE_SCALE = 1.15


class TextRenderer:
    """Renders arbitrary text as a handwriting-style image from a generated alphabet."""

    def __init__(self, alphabet: dict[str, list[GlyphBitmap]], config: dict | None = None):
        self.alphabet = alphabet
        cfg = config or load_config()
        self.cfg = cfg["renderer"]
        self.target_x_height = float(self.cfg["target_x_height_px"])
        self.rng = np.random.default_rng(7)
        self._warned_chars: set[str] = set()

        glyphs = [g for variants in alphabet.values() for g in variants]
        if not glyphs:
            raise ValueError("alphabet is empty")
        ascents, descents, advances = [], [], []
        for g in glyphs:
            k = self._scale(g)
            ascents.append(-g.metrics.bbox[1] * k)              # ink top above the baseline
            descents.append(-g.metrics.baseline_offset * k)     # ink bottom below the baseline
            advances.append(g.metrics.advance_width * k)
        self.ascent = max(1, math.ceil(max(ascents)))
        self.descent = max(0, math.ceil(max(descents)))
        self.mean_advance = float(np.mean(advances)) + self.cfg["char_spacing_px"]

    # -- glyph selection and scaling -----------------------------------------

    def _scale(self, glyph: GlyphBitmap) -> float:
        """Factor that brings this glyph to the common x-height."""
        return self.target_x_height / glyph.metrics.x_height

    def _pick_glyph(self, char: str) -> tuple[GlyphBitmap, float] | None:
        key = char.lower()
        if key in self.alphabet and self.alphabet[key]:
            variants = self.alphabet[key]
            glyph = variants[self.rng.integers(0, len(variants))]
            k = self._scale(glyph) * (UPPERCASE_SCALE if char.isupper() else 1.0)
            # No dedicated uppercase alphabet yet: approximate with an upscaled lowercase glyph
            # (documented limitation above). Scaling about the baseline keeps it on the line.
            return glyph, k
        if char not in self._warned_chars and not char.isspace():
            logger.warning("No glyph available for %r; rendering as blank space.", char)
            self._warned_chars.add(char)
        return None

    @staticmethod
    def _resize(image: np.ndarray, k: float) -> np.ndarray:
        h, w = image.shape[:2]
        size = (max(1, int(round(w * k))), max(1, int(round(h * k))))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)

    def _jitter_glyph(self, img: np.ndarray) -> tuple[np.ndarray, int]:
        """Apply small rotation jitter and return (image, vertical baseline offset)."""
        rot_deg = self.rng.uniform(-self.cfg["rotation_jitter_deg"], self.cfg["rotation_jitter_deg"])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
        jittered = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        baseline_offset = int(self.rng.integers(-self.cfg["baseline_jitter_px"], self.cfg["baseline_jitter_px"] + 1))
        return jittered, baseline_offset

    # -- layout -----------------------------------------------------------------

    def render_line(self, text: str) -> np.ndarray:
        """Render a single line of text (no newline handling). Every line has the same height and baseline."""
        tracking = self.cfg["char_spacing_px"]
        word_spacing = self.cfg["word_spacing_px"]
        margin = self.cfg["canvas_margin_px"]

        leading = int((self.ascent + self.descent) * (self.cfg["line_height_multiplier"] - 1.0))
        canvas_h = self.ascent + self.descent + leading + 2 * margin
        baseline_y = margin + leading // 2 + self.ascent

        placed = []  # (image, x, y_top) in canvas-relative coordinates before the left margin is applied
        pen_x = 0.0
        for ch in text:
            if ch == " ":
                pen_x += word_spacing
                continue
            picked = self._pick_glyph(ch)
            if picked is None:
                pen_x += self.mean_advance
                continue
            glyph, k = picked
            img, jitter_y = self._jitter_glyph(self._resize(glyph.image, k))
            m = glyph.metrics
            x = pen_x + m.bbox[0] * k                                   # pen origin + left side bearing
            y = baseline_y + m.bbox[1] * k + jitter_y                   # ink top = baseline + (negative) bbox.y
            placed.append((img, int(round(margin + x)), int(round(y))))
            pen_x += m.advance_width * k + tracking

        canvas_w = int(math.ceil(pen_x)) + 2 * margin
        canvas = np.zeros((canvas_h, max(canvas_w, 1)), dtype=np.uint8)
        for img, x, y in placed:
            self._paste_max(canvas, img, x, y)
        return canvas

    @staticmethod
    def _paste_max(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
        """Composite `img` onto `canvas` at (x, y) with per-pixel max, clipped to the canvas."""
        h, w = img.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return
        region = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = np.maximum(region, img[y0 - y:y1 - y, x0 - x:x1 - x])

    def render_text(self, text: str, max_line_width_px: int = 1400) -> np.ndarray:
        """Render possibly multi-line text, word-wrapping to `max_line_width_px`."""
        words = text.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            estimated_w = len(candidate) * self.mean_advance
            if estimated_w > max_line_width_px and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        if not lines:
            lines = [""]

        line_images = [self.render_line(line) for line in lines]
        width = max(img.shape[1] for img in line_images)
        total_height = sum(img.shape[0] for img in line_images)
        canvas = np.zeros((total_height, width), dtype=np.uint8)
        y = 0
        for img in line_images:
            h, w = img.shape[:2]
            canvas[y:y + h, 0:w] = img
            y += h
        return canvas

    def render_to_bgr_on_white(self, text: str) -> np.ndarray:
        """Render text and composite onto a white background as ink-colored (dark blue) strokes,
        for a natural "pen on paper" look in the UI/export.
        """
        gray_ink = self.render_text(text)
        h, w = gray_ink.shape[:2]
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        ink_color = np.array([120, 40, 20], dtype=np.uint8)  # dark blue-ish BGR "ink"
        alpha = (gray_ink.astype(np.float32) / 255.0)[..., None]
        canvas = (canvas.astype(np.float32) * (1 - alpha) + ink_color.astype(np.float32) * alpha).astype(np.uint8)
        return canvas
