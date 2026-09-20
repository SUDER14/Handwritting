"""CNN-based recognition backend (future_work.md: "implement
src/recognition/cnn_recognizer.py as a Recognizer subclass ... trained on
EMNIST-letters ... Register it in src/recognition/__init__.py::get_recognizer").

Unlike `LabeledRecognizer` (trusts the user's typed word) or
`TemplateMatchingRecognizer` (nearest-neighbour against a fixed printed
font), this backend loads a trained `CharCNN` (see `cnn_model.py`) and
classifies each segmented glyph crop from its pixels alone. That makes it
the first backend that actually *detects* what was written rather than
assuming it — `analyze_sample`/`recognize_with_cnn` in `src/pipeline.py`
use it to cross-check the "labeled" backend and to flag low-confidence
glyphs, which are frequently the symptom of under-segmented cursive joins
(see `src/segmentation/segmenter.py`'s touching-glyph splitter, the other
half of the cursive-handling fix).
"""
from __future__ import annotations

import functools
import string
from pathlib import Path

import cv2
import numpy as np
import torch

from src.utils.config import resolve_path

from .base import Recognizer
from .cnn_model import CharCNN

EMNIST_CANVAS = 28
DEFAULT_CHECKPOINT = "models/checkpoints/char_cnn.pt"
_DEFAULT_LABELS = list(string.ascii_lowercase)  # index 0..25 -> a..z (matches EMNISTLettersHF)


def normalize_glyph_for_cnn(glyph: np.ndarray, canvas: int = EMNIST_CANVAS, margin: float = 0.75) -> np.ndarray:
    """Resize/center a binary glyph crop (ink=255) to `canvas`x`canvas`, EMNIST-style.

    Mirrors the normalization the model was trained on: aspect-preserving
    resize so the glyph occupies `margin` of the canvas, centered on a black
    background — the same convention `TemplateMatchingRecognizer._normalize_glyph`
    uses at a different canvas size, kept separate here because it must match
    exactly what `scripts/train_cnn_recognizer.py` fed the model.
    """
    h, w = glyph.shape[:2]
    if h == 0 or w == 0 or glyph.max() == 0:
        return np.zeros((canvas, canvas), dtype=np.uint8)
    scale = (canvas * margin) / max(h, w)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    x0 = (canvas - new_w) // 2
    y0 = (canvas - new_h) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


@functools.lru_cache(maxsize=4)
def _load_checkpoint(checkpoint_path: str, device: str) -> tuple[CharCNN, tuple[str, ...]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained CNN recognizer checkpoint at {path}. "
            "Run `python scripts/train_cnn_recognizer.py` first."
        )
    ckpt = torch.load(path, map_location=device)
    labels = tuple(ckpt.get("labels", _DEFAULT_LABELS))
    model = CharCNN(num_classes=len(labels)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, labels


class CNNRecognizer(Recognizer):
    """Loads a trained `CharCNN` checkpoint and classifies glyphs a-z."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu", **_ignored):
        # `_ignored` swallows kwargs other backends accept (e.g. `label_text`)
        # so `get_recognizer(backend, **shared_kwargs)` can pass one kwargs
        # dict without every backend needing to accept every key.
        resolved = Path(checkpoint_path) if checkpoint_path else resolve_path(DEFAULT_CHECKPOINT)
        self.checkpoint_path = resolved
        self.device = device
        self.model, self.labels = _load_checkpoint(str(resolved), device)
        self.last_confidences: list[float] = []

    @torch.no_grad()
    def recognize(self, glyph_images: list[np.ndarray]) -> list[str]:
        if not glyph_images:
            self.last_confidences = []
            return []
        batch = np.stack([normalize_glyph_for_cnn(g) for g in glyph_images]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).unsqueeze(1).to(self.device)  # N,1,28,28
        probs = torch.softmax(self.model(tensor), dim=1)
        conf, idx = probs.max(dim=1)
        self.last_confidences = conf.cpu().tolist()
        return [self.labels[i] for i in idx.cpu().tolist()]
