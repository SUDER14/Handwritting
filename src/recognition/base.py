"""Pluggable character-recognition interface.

Character identification is deliberately decoupled from segmentation so the
backend can be swapped without touching any other module — see config key
`recognition.backend`. Three backends are provided:

- LabeledRecognizer: uses ground-truth text supplied by the user for their
  own sample (the MVP default — see docs/limitations.md for why this is the
  practical choice rather than solving open-vocabulary handwritten character
  recognition from scratch).
- TemplateMatchingRecognizer: nearest-neighbour against per-class template
  glyphs, a classic non-learned baseline.
- CNNRecognizer (src/recognition/cnn_recognizer.py): a CharCNN trained on
  EMNIST-letters (scripts/train_cnn_recognizer.py) that classifies each
  glyph from its pixels — the first backend that genuinely detects what was
  written rather than assuming it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Recognizer(ABC):
    """Abstract base class every recognition backend must implement."""

    @abstractmethod
    def recognize(self, glyph_images: list[np.ndarray]) -> list[str]:
        """Map each glyph image (binary, ink=255) to a predicted character.

        Args:
            glyph_images: list of cropped binary glyph images, left-to-right order.

        Returns:
            List of predicted characters, same length and order as input.
        """
        raise NotImplementedError
