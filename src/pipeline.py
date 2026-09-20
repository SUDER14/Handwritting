"""End-to-end orchestration: image -> segmented/labeled glyphs -> style profile
-> generated alphabet -> renderer.

This is the module the Streamlit app (app/app.py) and the end-to-end test
(tests/test_end_to_end.py) both call into, so the "real" pipeline logic
lives in one tested place rather than being duplicated in the UI.

Two generation backends are supported side by side (see docs/experiments.md
for the baseline-vs-proposed comparison):
    "baseline" -> src/generator/baseline_generator.py (CV style-transform)
    "neural"   -> src/style_encoder/inference.py (trained conditional VAE)
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

from src.features.style_extractor import StyleFeatureExtractor, StyleProfile
from src.generator.baseline_generator import BaselineGlyphGenerator
from src.generator.glyph import GlyphBitmap, GlyphPriors, bitmap_from_generated, make_bitmap, writer_spacing_ratio
from src.preprocessing.pipeline import Preprocessor, PreprocessingResult
from src.recognition import get_recognizer
from src.segmentation.segmenter import Segmenter, SegmentationResult
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SampleAnalysis:
    preprocessing: PreprocessingResult
    segmentation: SegmentationResult
    chars: list[str]
    observed_glyphs: dict[str, np.ndarray]   # char -> raw segmented crop (binary, native scale)
    style_profile: StyleProfile
    observed_bitmaps: dict[str, GlyphBitmap]  # same crops wrapped with GlyphMetrics measured from this sample


def analyze_sample(
    image_bgr: np.ndarray,
    label_text: str,
    config: dict | None = None,
    recognition_backend: str | None = None,
) -> SampleAnalysis:
    """Run preprocessing -> segmentation -> recognition -> style-feature extraction.

    `label_text` is the word the user says they wrote (e.g. "quick"). By
    default (`recognition_backend=None`, or "labeled") it's trusted as
    ground truth and used both to reconcile segmentation
    (`segment_and_align_to_label`) and to assign characters directly — the
    reliable path the alphabet-generation pipeline depends on for correct
    per-character identity.

    Passing `recognition_backend="cnn"` instead switches to open-vocabulary
    detection: segmentation is *not* forced to match `label_text`'s length
    (`segment()`, not `segment_and_align_to_label()`), and
    `CNNRecognizer` (src/recognition/cnn_recognizer.py) predicts each
    character from pixels alone. This is what actually answers "what does
    the model think was written" rather than assuming the typed label is
    correct — useful when the label may be wrong, or to verify detection
    quality — but a misrecognized character mis-keys `observed_glyphs`, so
    it's opt-in rather than the default for the generation pipeline. See
    also `recognize_with_cnn` for a non-destructive cross-check that runs
    the CNN alongside the "labeled" backend without affecting `chars`.
    """
    cfg = config or load_config()
    backend = recognition_backend or cfg["recognition"]["backend"]

    pre = Preprocessor(cfg)
    # The label length tells preprocessing whether there is enough text to estimate skew at all
    # (see preprocessing.deskew_min_chars); with no label it falls back to counting components.
    pre_result = pre.process(image_bgr=image_bgr, n_chars=len(label_text) if backend == "labeled" else None)

    seg = Segmenter(cfg)
    if backend == "labeled":
        seg_result = seg.segment_and_align_to_label(pre_result.cropped, label_text)
    else:
        seg_result = seg.segment(pre_result.cropped)

    recognizer = get_recognizer(
        backend, label_text=label_text, checkpoint_path=str(resolve_path(cfg["recognition"]["cnn_checkpoint_path"]))
    )
    chars = recognizer.recognize([g.image for g in seg_result.glyphs])

    observed_glyphs: dict[str, np.ndarray] = {}
    for char, glyph in zip(chars, seg_result.glyphs):
        # If a character repeats in the sample (e.g. "quick" has no repeats, but
        # "hello" has two 'l's), keep the first occurrence for style extraction.
        observed_glyphs.setdefault(char.lower(), glyph.image)

    priors = GlyphPriors.from_config(cfg)
    extractor = StyleFeatureExtractor(priors)
    style_profile = extractor.extract_style_profile(
        chars, seg_result.glyphs, seg_result.baseline_y, seg_result.x_height_top_y
    )

    # Metrics for the real glyphs: x-height and baseline are the writer's, measured on this line, so
    # descender letters get a negative baseline_offset relative to the SAME baseline as everything else.
    line = style_profile.line_metrics
    spacing = writer_spacing_ratio(style_profile.mean_char_spacing_px, line.x_height, priors) * line.x_height
    observed_bitmaps: dict[str, GlyphBitmap] = {}
    for char, glyph in zip(chars, seg_result.glyphs):
        if char.lower() in observed_bitmaps:
            continue
        _, y, _, h = glyph.bbox
        observed_bitmaps[char.lower()] = make_bitmap(glyph.image, line.x_height, line.baseline_y - y, spacing)

    return SampleAnalysis(
        preprocessing=pre_result,
        segmentation=seg_result,
        chars=chars,
        observed_glyphs=observed_glyphs,
        style_profile=style_profile,
        observed_bitmaps=observed_bitmaps,
    )


@dataclass
class RecognitionCheck:
    """Per-glyph comparison of the typed label against the CNN's own reading."""

    typed_chars: list[str]
    predicted_chars: list[str]
    confidences: list[float]
    agreement: list[bool]
    accuracy: float


def recognize_with_cnn(analysis: SampleAnalysis, config: dict | None = None) -> RecognitionCheck | None:
    """Cross-check `analysis.chars` (from the "labeled" backend) against the
    trained CNN's own, independent reading of the same glyph crops.

    Non-destructive: never mutates `analysis`, so it's safe to call after the
    default "labeled" pipeline just to surface how confidently the model
    agrees with the typed label. Low agreement/confidence on a glyph is
    frequently the symptom of a segmentation problem (e.g. a touching
    cursive join that still spans two letters after splitting) rather than
    the classifier being wrong — see src/segmentation/segmenter.py. Returns
    None if no CNN checkpoint has been trained yet
    (scripts/train_cnn_recognizer.py).
    """
    from src.recognition.cnn_recognizer import CNNRecognizer

    cfg = config or load_config()
    checkpoint_path = resolve_path(cfg["recognition"]["cnn_checkpoint_path"])
    if not checkpoint_path.exists():
        return None

    recognizer = CNNRecognizer(checkpoint_path=str(checkpoint_path))
    glyph_images = [g.image for g in analysis.segmentation.glyphs]
    predicted = recognizer.recognize(glyph_images)
    confidences = list(recognizer.last_confidences)
    typed = [c.lower() for c in analysis.chars]
    agreement = [t == p for t, p in zip(typed, predicted)]
    accuracy = float(np.mean(agreement)) if agreement else 0.0

    return RecognitionCheck(
        typed_chars=typed, predicted_chars=predicted, confidences=confidences,
        agreement=agreement, accuracy=accuracy,
    )


def generate_alphabet_baseline(
    analysis: SampleAnalysis, config: dict | None = None, charset: str = string.ascii_lowercase
) -> dict[str, list[GlyphBitmap]]:
    cfg = config or load_config()
    generator = BaselineGlyphGenerator(cfg)
    return generator.generate_alphabet(analysis.observed_bitmaps, analysis.style_profile, charset=charset)


def generate_alphabet_neural(
    analysis: SampleAnalysis, checkpoint_path: str | None = None, charset: str = string.ascii_lowercase,
    config: dict | None = None,
) -> dict[str, list[GlyphBitmap]]:
    """Requires a trained checkpoint (python scripts/train_style_encoder.py).

    The VAE emits 28x28 glyphs with no absolute size information; they are wrapped with `GlyphMetrics`
    from typographic priors (`bitmap_from_generated`) so the renderer can put them at the same x-height
    and baseline as the real, native-resolution observed glyphs.
    """
    from src.style_encoder.inference import StyleEncoderInference  # lazy import: torch is heavy

    cfg = config or load_config()
    priors = GlyphPriors.from_config(cfg)
    style = analysis.style_profile
    ratio = writer_spacing_ratio(style.mean_char_spacing_px, style.x_height_px, priors)

    engine = StyleEncoderInference(checkpoint_path=checkpoint_path)
    writer_style = engine.encode_writer_style(analysis.observed_glyphs)
    raw = engine.generate_alphabet(writer_style, charset=charset, variants=3)
    alphabet = {c: [bitmap_from_generated(c, img, ratio, priors) for img in imgs] for c, imgs in raw.items()}

    # Hybrid policy (section 25): prefer the real observed glyph over the
    # generated one for characters actually present in the sample.
    for char, bitmap in analysis.observed_bitmaps.items():
        if char in alphabet:
            alphabet[char][0] = bitmap
    return alphabet
