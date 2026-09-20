"""Ground-truth-labeled recognition backend (MVP default).

The user already knows what word they wrote (e.g. "quick"), and
`Segmenter.segment_and_align_to_label` already reconciles the number of
segmented glyphs to `len(label_text)`. So identification here is just
zipping glyphs to characters in left-to-right order — trivially correct
given correct segmentation, and it sidesteps the much harder unconstrained
handwriting-recognition problem for the MVP. This is a placeholder for
`CNNRecognizer` (see base.py), not a permanent design decision.
"""
from __future__ import annotations

import numpy as np

from .base import Recognizer


class LabeledRecognizer(Recognizer):
    """Assigns characters from a known ground-truth string, in order."""

    def __init__(self, label_text: str):
        self.label_text = label_text

    def recognize(self, glyph_images: list[np.ndarray]) -> list[str]:
        if len(glyph_images) != len(self.label_text):
            raise ValueError(
                f"Glyph count ({len(glyph_images)}) does not match label length "
                f"({len(self.label_text)}). Segmentation should have reconciled this "
                "via segment_and_align_to_label()."
            )
        return list(self.label_text)
