"""Factory for pluggable recognition backends (see config key `recognition.backend`)."""
from __future__ import annotations

from .base import Recognizer
from .cnn_recognizer import CNNRecognizer
from .labeled import LabeledRecognizer
from .template_matching import TemplateMatchingRecognizer

__all__ = [
    "Recognizer", "LabeledRecognizer", "TemplateMatchingRecognizer", "CNNRecognizer", "get_recognizer",
]


def get_recognizer(backend: str, **kwargs) -> Recognizer:
    """Instantiate a recognition backend by name.

    Args:
        backend: one of "labeled", "template", "cnn".
        kwargs: backend-specific constructor args (e.g. `label_text=` for "labeled",
            `checkpoint_path=` for "cnn"). Unrecognized kwargs are ignored by
            backends that don't need them, so one shared kwargs dict can be
            passed regardless of which backend is selected.
    """
    if backend == "labeled":
        return LabeledRecognizer(label_text=kwargs.get("label_text", ""))
    if backend == "template":
        return TemplateMatchingRecognizer(**{k: v for k, v in kwargs.items() if k == "charset"})
    if backend == "cnn":
        return CNNRecognizer(**{k: v for k, v in kwargs.items() if k in ("checkpoint_path", "device")})
    raise ValueError(f"Unknown recognition backend: {backend!r}")
