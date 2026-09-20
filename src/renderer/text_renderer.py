"""Text-to-handwriting rendering engine (section 14 of the design brief).

Composes arbitrary text into an image using a personalized alphabet,
with natural (non-identical) variation: per-occurrence glyph-variant
selection, small baseline jitter, and slight rotation jitter — real
handwriting is never pixel-identical letter to letter.

CURRENT SCOPE (documented limitation, see docs/limitations.md): the
active alphabet is lowercase a-z (per project priority). Uppercase
letters fall back to an upscaled lowercase glyph if no uppercase
alphabet has been generated; punctuation/digits are rendered as
whitespace-width gaps for now — both are architected as drop-in
extensions (section 31) once uppercase/digit alphabets are generated
the same way as lowercase.
"""
from __future__ import annotations

import string

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class TextRenderer:
    """Renders arbitrary text as a handwriting-style image from a generated alphabet."""

    def __init__(self, alphabet: dict[str, list[np.ndarray]], config: dict | None = None):
        self.alphabet = alphabet
        cfg = config or load_config()
        self.cfg = cfg["renderer"]
        self.rng = np.random.default_rng(7)
        self._warned_chars: set[str] = set()

        heights = [imgs[0].shape[0] for imgs in alphabet.values() if imgs]
        widths = [imgs[0].shape[0] for imgs in alphabet.values() if imgs]
        self.mean_height = int(np.mean(heights)) if heights else 40
        self.mean_char_width = int(np.mean(widths)) if widths else 30

    def _pick_glyph(self, char: str) -> np.ndarray | None:
        key = char.lower()
        if key in self.alphabet and self.alphabet[key]:
            variants = self.alphabet[key]
            img = variants[self.rng.integers(0, len(variants))]
            if char.isupper():
                # No dedicated uppercase alphabet yet: approximate with an
                # upscaled lowercase glyph (documented limitation above).
                scale = 1.15
                h, w = img.shape[:2]
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            return img
        if char not in self._warned_chars and not char.isspace():
            logger.warning("No glyph available for %r; rendering as blank space.", char)
            self._warned_chars.add(char)
        return None

    def _jitter_glyph(self, img: np.ndarray) -> tuple[np.ndarray, int]:
        """Apply small rotation jitter and return (image, vertical baseline offset)."""
        rot_deg = self.rng.uniform(-self.cfg["rotation_jitter_deg"], self.cfg["rotation_jitter_deg"])
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
        jittered = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        baseline_offset = int(self.rng.integers(-self.cfg["baseline_jitter_px"], self.cfg["baseline_jitter_px"] + 1))
        return jittered, baseline_offset

    def render_line(self, text: str) -> np.ndarray:
        """Render a single line of text (no newline handling) to a tight-cropped image."""
        char_spacing = self.cfg["char_spacing_px"]
        word_spacing = self.cfg["word_spacing_px"]
        margin = self.cfg["canvas_margin_px"]

        placed = []  # list of (img, x, y)
        cursor_x = 0
        max_h = self.mean_height

        for ch in text:
            if ch == " ":
                cursor_x += word_spacing
                continue
            img = self._pick_glyph(ch)
            if img is None:
                cursor_x += self.mean_char_width + char_spacing
                continue
            img, baseline_offset = self._jitter_glyph(img)
            h, w = img.shape[:2]
            max_h = max(max_h, h)
            placed.append((img, cursor_x, baseline_offset))
            cursor_x += w + char_spacing

        canvas_w = cursor_x + 2 * margin
        canvas_h = int(max_h * self.cfg["line_height_multiplier"]) + 2 * margin
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        baseline_y = margin + int(max_h * 1.0)

        for img, x, baseline_offset in placed:
            h, w = img.shape[:2]
            y = baseline_y - h + baseline_offset
            y = max(0, min(canvas_h - h, y))
            x = max(0, min(canvas_w - w, margin + x))
            region = canvas[y:y + h, x:x + w]
            canvas[y:y + h, x:x + w] = np.maximum(region, img)

        return canvas

    def render_text(self, text: str, max_line_width_px: int = 1400) -> np.ndarray:
        """Render possibly multi-line text, word-wrapping to `max_line_width_px`."""
        words = text.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            estimated_w = len(candidate) * (self.mean_char_width + self.cfg["char_spacing_px"])
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
        mask = gray_ink > 20
        alpha = (gray_ink.astype(np.float32) / 255.0)[..., None]
        canvas = (canvas.astype(np.float32) * (1 - alpha) + ink_color.astype(np.float32) * alpha).astype(np.uint8)
        return canvas
