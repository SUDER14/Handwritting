"""Generation backends behind ONE interface for the evaluation protocol (src/evaluation/protocol.py):

    backend.generate(refs: list[Ref], text: str) -> uint8 gray line image, dark ink on light paper (or None)

* BaselineBackend -- the font-glyph baseline (warped system font, real crops for letters seen in the references).
* NeuralBackend   -- the EMNIST glyph VAE (ablation row only; no further development, per ROADMAP).
* (Stage 3) VATr wrapper lives in src/generator/vatr_backend.py and implements the same interface.

With K > 1 references, each reference line is analysed separately and their observed glyph bitmaps are merged
(first reference wins for a letter both contain; later references fill in letters the earlier ones lack); the style
profile comes from the first reference. `coverage` accumulates, over all generated targets, how many target letters were
present in the merged references (=> real glyph used) and how many were absent (=> generated / font fallback):
that is the Task 2.4 per-character coverage statistic.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from src.evaluation.harness import letters_only, to_binary_ink
from src.pipeline import analyze_sample, generate_alphabet_baseline, generate_alphabet_neural
from src.renderer.text_renderer import TextRenderer


class _PipelineBackend:
    name = "pipeline"

    def __init__(self, config: dict):
        self.cfg = config
        self.coverage_seen: Counter = Counter()      # target letters found in the references
        self.coverage_missing: Counter = Counter()   # target letters absent from the references (generated/fallback)

    def _analysis(self, refs):
        first, merged = None, {}
        for r in refs:
            bgr = cv2.cvtColor(r.gray, cv2.COLOR_GRAY2BGR) if r.gray.ndim == 2 else r.gray
            a = analyze_sample(bgr, letters_only(r.text), self.cfg)
            first = first or a
            for ch, bm in a.observed_bitmaps.items():
                merged.setdefault(ch, bm)
        first.observed_bitmaps = merged
        return first

    def _alphabet(self, analysis):
        raise NotImplementedError

    def generate(self, refs, text: str):
        analysis = self._analysis(refs)
        for ch in text.replace(" ", ""):
            (self.coverage_seen if ch in analysis.observed_bitmaps else self.coverage_missing)[ch] += 1
        rendered = to_binary_ink(TextRenderer(self._alphabet(analysis), self.cfg).render_line(text))   # ink = 255 on 0
        return (255 - rendered).astype(np.uint8)

    def coverage(self) -> dict:
        seen, miss = sum(self.coverage_seen.values()), sum(self.coverage_missing.values())
        per_char = {c: {"seen": self.coverage_seen[c], "missing": self.coverage_missing[c]}
                    for c in sorted(set(self.coverage_seen) | set(self.coverage_missing))}
        return {"letters_in_references": seen, "letters_absent": miss,
                "fallback_rate": miss / max(seen + miss, 1), "per_char": per_char}


class BaselineBackend(_PipelineBackend):
    name = "baseline"

    def _alphabet(self, analysis):
        return generate_alphabet_baseline(analysis, self.cfg)


class NeuralBackend(_PipelineBackend):
    name = "glyph_vae"

    def __init__(self, config: dict, checkpoint_path: str):
        super().__init__(config)
        self.checkpoint_path = checkpoint_path

    def _alphabet(self, analysis):
        return generate_alphabet_neural(analysis, checkpoint_path=self.checkpoint_path, config=self.cfg)
