"""Template-matching recognition backend (non-learned baseline).

Renders each candidate character from a system font into a normalized
bitmap template, then classifies an unknown glyph as the template it
correlates with most strongly (normalized cross-correlation on
size-normalized, centered binary images). This is the classic
"template matching for prototype" option from the design brief: useful as
a sanity-check baseline and as a fallback when no ground-truth label is
available, but it is expected to be less accurate than a trained CNN
(see `CNNRecognizer`, a future pluggable upgrade — this class implements
the same `Recognizer` interface so swapping backends requires no changes
elsewhere).
"""
from __future__ import annotations

import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .base import Recognizer

DEFAULT_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoesc.ttf",
    "C:/Windows/Fonts/comic.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
CANVAS = 64


def _load_font(size: int = 48) -> ImageFont.FreeTypeFont:
    for path in DEFAULT_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_char_template(char: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Render a character to a centered CANVASxCANVAS binary (ink=255) template."""
    img = Image.new("L", (CANVAS, CANVAS), color=0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - w) / 2 - bbox[0]
    y = (CANVAS - h) / 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return np.array(img)


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    """Resize a binary glyph crop to CANVASxCANVAS, preserving aspect ratio, centered."""
    h, w = glyph.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    scale = (CANVAS * 0.8) / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    x0 = (CANVAS - new_w) // 2
    y0 = (CANVAS - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


class TemplateMatchingRecognizer(Recognizer):
    """Nearest-neighbour recognizer using rendered font glyphs as class templates."""

    def __init__(self, charset: str = string.ascii_lowercase + string.ascii_uppercase):
        self.charset = charset
        font = _load_font()
        self.templates: dict[str, np.ndarray] = {
            c: _render_char_template(c, font) for c in charset
        }

    def _best_match(self, glyph_norm: np.ndarray) -> str:
        glyph_f = glyph_norm.astype(np.float32)
        best_char, best_score = None, -np.inf
        for char, template in self.templates.items():
            template_f = template.astype(np.float32)
            # Normalized cross-correlation (Pearson correlation of pixel intensities).
            gm, tm = glyph_f - glyph_f.mean(), template_f - template_f.mean()
            denom = (np.linalg.norm(gm) * np.linalg.norm(tm)) + 1e-8
            score = float((gm * tm).sum() / denom)
            if score > best_score:
                best_score, best_char = score, char
        return best_char or "?"

    def recognize(self, glyph_images: list[np.ndarray]) -> list[str]:
        return [self._best_match(_normalize_glyph(g)) for g in glyph_images]
