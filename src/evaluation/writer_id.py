"""Writer-identity embedding interface, used for the *style fidelity* metric.

Style fidelity = cosine distance between the writer embedding of a GENERATED
line image and that of a REFERENCE line image from the same writer. Lower is
better. It is only as meaningful as the embedder, so every result records which
embedder produced it and whether that embedder is a stub.

A real embedder is a model trained to make same-writer images close and
different-writer images far (e.g. a metric-learning CNN on IAM train-split
writers). **It does not exist yet.** Until it does, `RuleBasedStubEmbedder`
keeps the harness end-to-end runnable; its numbers are a placeholder, flagged
`is_stub=True` in metrics.json and results/index.csv, and must not be reported
as writer-identity results.

To plug in the trained model: subclass `WriterEmbedder`, set `name` and
`is_stub = False`, implement `embed`, and register it in `get_writer_embedder`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.features.style_extractor import StyleFeatureExtractor
from src.segmentation.segmenter import Segmenter


class WriterEmbedder(ABC):
    """Maps a line image to a fixed-length writer-identity vector."""

    name: str = "unnamed"
    is_stub: bool = True   # subclasses that are real trained models must set this to False

    @abstractmethod
    def embed(self, line_image: np.ndarray) -> np.ndarray:
        """Embed one line image.

        Args:
            line_image: 2-D uint8, **ink = 255 on background = 0** (the pipeline's canonical
                binary convention), any width. Callers binarise real photos first
                (`Preprocessor.process(...).cropped`).

        Returns:
            1-D float array (fixed length for a given embedder).
        """
        raise NotImplementedError

    def distance(self, image_a: np.ndarray, image_b: np.ndarray) -> float:
        return cosine_distance(self.embed(image_a), self.embed(image_b))

    def describe(self) -> dict:
        return {"name": self.name, "is_stub": self.is_stub}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity, in [0, 2]. Raises on a zero vector rather than inventing a value."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine distance is undefined for a zero embedding (empty/blank image?)")
    return float(1.0 - np.dot(a, b) / (na * nb))


class RuleBasedStubEmbedder(WriterEmbedder):
    """PLACEHOLDER. Scale-free hand-crafted glyph statistics, not a writer-ID model.

    Vector: [mean aspect, mean slant/45, mean stroke/height, mean ink density,
    mean curvature, std slant/45] over the segmented glyphs. All components are
    dimensionless so a 64-px and a 400-px image are comparable. It is NOT
    centred or whitened, so cosine distances between any two handwriting images
    will be small and weakly discriminative -- one of the reasons this is a stub.
    """

    name = "stub_rule_based_v0"
    is_stub = True

    def __init__(self, config: dict | None = None):
        self._segmenter = Segmenter(config)
        self._extractor = StyleFeatureExtractor()

    def embed(self, line_image: np.ndarray) -> np.ndarray:
        binary = ((np.asarray(line_image) > 127).astype(np.uint8)) * 255
        glyphs = self._segmenter.segment(binary).glyphs
        if not glyphs:
            raise ValueError("no glyphs found in image; cannot embed")
        feats = [self._extractor.extract_glyph_features("?", g.image) for g in glyphs]
        slants = np.array([f.slant_deg for f in feats]) / 45.0
        return np.array([
            np.mean([f.aspect_ratio for f in feats]),
            np.mean(slants),
            np.mean([f.stroke_width_px / max(1, f.height_px) for f in feats]),
            np.mean([f.ink_density for f in feats]),
            np.mean([f.curvature_score for f in feats]),
            np.std(slants),
        ], dtype=np.float64)


def get_writer_embedder(name: str = "stub", config: dict | None = None) -> WriterEmbedder:
    """Factory keyed by `evaluation.writer_embedder` in config.yaml."""
    if name == "stub":
        return RuleBasedStubEmbedder(config)
    raise NotImplementedError(
        f"Writer embedder {name!r} is not available. Only 'stub' exists; a trained writer-ID model must be "
        "implemented as a WriterEmbedder subclass and registered here (src/evaluation/writer_id.py)."
    )
