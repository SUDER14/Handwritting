"""Writer-identity embedding interface, used for the *style fidelity* metric.

Style fidelity = cosine distance between the writer embedding of a GENERATED
line image and that of a REFERENCE line image from the same writer. Lower is
better. It is only as meaningful as the embedder, so every result records which
embedder produced it and whether that embedder is a stub.

`TrainedWriterEmbedder` is the real one: the ResNet trained from scratch on IAM train-split
writers by scripts/train_writer_id.py (src/writer_id/). It is only a usable metric if its
retrieval accuracy on UNSEEN test writers clears `writer_id.eval.min_usable_top1`; `describe()`
reports that as `usable`, evaluate.py prints a loud warning when it is False, and it is recorded
in metrics.json / results/index.csv.

`RuleBasedStubEmbedder` remains available (`--writer-embedder stub`) for smoke tests only; its
numbers are placeholders, flagged `is_stub=True`, and must never be reported.
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


class TrainedWriterEmbedder(WriterEmbedder):
    """The from-scratch writer-ID ResNet from a run directory (models/checkpoints/<run_id>/)."""

    is_stub = False

    def __init__(self, run_dir, config: dict | None = None):
        from src.utils.config import load_config
        from src.writer_id.runs import best_checkpoint, load_encoder, read_test_metrics

        self.run_dir = run_dir
        self.encoder, self.run_config = load_encoder(run_dir)
        wid = self.run_config["writer_id"]
        self._height, self._eval_cfg = wid["input_height"], wid["eval"]
        self.run_id = self.run_config["run_id"]
        self.name = f"writer_id:{self.run_id}"
        self.checkpoint = best_checkpoint(run_dir).name
        self.test_metrics = read_test_metrics(run_dir)
        self.min_usable_top1 = float((config or load_config())["writer_id"]["eval"]["min_usable_top1"])

    @property
    def usable(self) -> bool:
        """True only if the run recorded test-writer top-1 retrieval >= the usability threshold."""
        return self.test_metrics is not None and self.test_metrics["top1"] >= self.min_usable_top1

    def embed(self, line_image: np.ndarray) -> np.ndarray:
        from src.writer_id.data import canonical_from_binary
        from src.writer_id.retrieval import embed_line

        if not np.any(np.asarray(line_image) > 127):
            raise ValueError("blank image; cannot embed")
        return embed_line(self.encoder, canonical_from_binary(line_image, self._height), self._eval_cfg).astype(np.float64)

    def describe(self) -> dict:
        tm = self.test_metrics or {}
        return {"name": self.name, "is_stub": False, "run_id": self.run_id, "checkpoint": self.checkpoint,
                "test_top1": tm.get("top1"), "test_map": tm.get("map"), "test_chance_top1": tm.get("chance_top1"),
                "min_usable_top1": self.min_usable_top1, "usable": self.usable}


def get_writer_embedder(name: str = "trained", config: dict | None = None, run_dir=None) -> WriterEmbedder:
    """Factory keyed by `evaluation.writer_embedder` in config.yaml ("trained" | "stub").

    "trained" uses `run_dir`, else `evaluation.writer_id_run`, else the newest run under models/checkpoints/.
    It raises FileNotFoundError when no trained run exists -- it never silently falls back to the stub.
    """
    if name == "stub":
        return RuleBasedStubEmbedder(config)
    if name == "trained":
        from pathlib import Path

        from src.utils.config import resolve_path
        from src.writer_id.runs import latest_run

        chosen = run_dir or (config or {}).get("evaluation", {}).get("writer_id_run")
        path = Path(chosen) if chosen else None
        if path is not None and not path.is_absolute():
            path = resolve_path(path)
        path = path or latest_run()
        if path is None:
            raise FileNotFoundError(
                "No trained writer-ID checkpoint found under models/checkpoints/. Train one with "
                "scripts/train_writer_id.py, pass --writer-id-run, or choose --writer-embedder stub explicitly "
                "(smoke tests only)."
            )
        return TrainedWriterEmbedder(path, config)
    raise NotImplementedError(f"Unknown writer embedder {name!r}; expected 'trained' or 'stub'.")
