"""Leave-one-out SSIM: hide a character, regenerate it, score it against the real crop.

The scoring path must not be able to see the held-out crop. The previous
implementation (scripts/evaluate.py) rebuilt only the *neural* writer embedding
from the remaining glyphs, but scored the baseline with the style profile of the
FULL sample -- whose height/width/slant/stroke averages include the held-out
glyph. Here the profile is always rebuilt from the remaining characters only
(`StyleFeatureExtractor.extract_style_profile_excluding`).
"""
from __future__ import annotations

import string

import numpy as np

from src.evaluation.metrics import ssim_against_reference
from src.features.style_extractor import StyleFeatureExtractor, StyleProfile
from src.generator.baseline_generator import BaselineGlyphGenerator
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def style_profile_without(analysis, held_out: str) -> StyleProfile:
    """The style profile used to score `held_out`: built from all other characters only."""
    return StyleFeatureExtractor().extract_style_profile_excluding(
        analysis.chars, analysis.segmentation.glyphs, held_out
    )


def leave_one_out_ssim(analysis, config: dict, backend: str = "baseline", engine=None) -> dict[str, float]:
    """SSIM per held-out character. Empty dict if fewer than two distinct letters were observed.

    Args:
        analysis: `SampleAnalysis` (needs .chars, .observed_glyphs, .segmentation.glyphs).
        backend: "baseline" or "neural". For "neural", pass a loaded `StyleEncoderInference` as `engine`.
    """
    candidates = sorted(c for c in set(analysis.chars) if c in string.ascii_lowercase and c in analysis.observed_glyphs)
    if len(candidates) < 2:
        logger.warning("Need at least 2 distinct observed letters for leave-one-out; skipping.")
        return {}
    if backend == "neural" and engine is None:
        raise ValueError("backend='neural' requires a loaded StyleEncoderInference as `engine`")

    scores: dict[str, float] = {}
    for held_out in candidates:
        if backend == "neural":
            remaining = {c: g for c, g in analysis.observed_glyphs.items() if c != held_out}
            generated = engine.generate_char(held_out, engine.encode_writer_style(remaining))
        else:
            style = style_profile_without(analysis, held_out)
            generated = BaselineGlyphGenerator(config).generate_unobserved(held_out, style)[0]
        scores[held_out] = ssim_against_reference(np.asarray(generated), analysis.observed_glyphs[held_out])
    return scores
