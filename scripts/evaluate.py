"""Runs the baseline-vs-proposed comparison described in docs/experiments.md
against a handwriting sample, including the leave-one-out SSIM protocol.

Usage:
    python scripts/evaluate.py --image data/samples/quick_sample.png --label quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.evaluation.metrics import (
    character_recognition_accuracy,
    ssim_against_reference,
    writer_style_similarity,
)
from src.generator.baseline_generator import BaselineGlyphGenerator
from src.pipeline import analyze_sample, generate_alphabet_baseline
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def leave_one_out_ssim(image_bgr: np.ndarray, label: str, cfg: dict, use_neural: bool) -> dict[str, float]:
    """For each observed character, regenerate it from the OTHER observed characters'
    style and score it against its real withheld glyph via SSIM.
    """
    scores = {}
    unique_chars = sorted(set(label))
    if len(unique_chars) < 2:
        logger.warning("Need at least 2 distinct characters for leave-one-out; skipping.")
        return scores

    full_analysis = analyze_sample(image_bgr, label, cfg)

    for held_out in unique_chars:
        remaining_glyphs = {c: g for c, g in full_analysis.observed_glyphs.items() if c != held_out}
        if not remaining_glyphs:
            continue

        if use_neural:
            from src.style_encoder.inference import StyleEncoderInference

            engine = StyleEncoderInference()
            writer_style = engine.encode_writer_style(remaining_glyphs)
            generated = engine.generate_char(held_out, writer_style)
        else:
            generator = BaselineGlyphGenerator(cfg)
            generated = generator.generate_unobserved(held_out, full_analysis.style_profile)[0]

        real = full_analysis.observed_glyphs[held_out]
        scores[held_out] = ssim_against_reference(generated, real)

    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--skip-neural", action="store_true", help="Skip neural backend (no trained checkpoint).")
    args = parser.parse_args()

    cfg = load_config()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    label = "".join(c for c in args.label.lower() if c.isalpha())

    analysis = analyze_sample(image_bgr, label, cfg)
    logger.info("=" * 70)
    logger.info("Sample: %r -> detected chars: %s", label, analysis.chars)

    cnn_recognizer = None
    cnn_checkpoint_path = resolve_path(cfg["recognition"]["cnn_checkpoint_path"])
    if cnn_checkpoint_path.exists():
        from src.recognition.cnn_recognizer import CNNRecognizer

        cnn_recognizer = CNNRecognizer(checkpoint_path=str(cnn_checkpoint_path))
    else:
        logger.info("No CNN recognizer checkpoint at %s; recognition-accuracy OCR proxy "
                     "will use template matching only.", cnn_checkpoint_path)

    logger.info("-" * 70)
    logger.info("BASELINE generator")
    baseline_alphabet = generate_alphabet_baseline(analysis, cfg)
    baseline_rec = character_recognition_accuracy(baseline_alphabet)
    baseline_sim = writer_style_similarity(baseline_alphabet, analysis.observed_glyphs)
    logger.info("  Recognition accuracy (template OCR proxy): %.1f%%", baseline_rec["__overall__"] * 100)
    if cnn_recognizer is not None:
        baseline_rec_cnn = character_recognition_accuracy(baseline_alphabet, recognizer=cnn_recognizer)
        logger.info("  Recognition accuracy (trained CNN OCR proxy): %.1f%%", baseline_rec_cnn["__overall__"] * 100)
    logger.info("  Writer-style similarity (cosine, rule-based features): %.3f", baseline_sim)
    loo_baseline = leave_one_out_ssim(image_bgr, label, cfg, use_neural=False)
    if loo_baseline:
        logger.info("  Leave-one-out SSIM per char: %s", {k: round(v, 3) for k, v in loo_baseline.items()})
        logger.info("  Leave-one-out SSIM mean: %.3f", float(np.mean(list(loo_baseline.values()))))

    checkpoint_path = resolve_path(cfg["training"]["checkpoint_path"])
    if not args.skip_neural and checkpoint_path.exists():
        logger.info("-" * 70)
        logger.info("NEURAL generator (conditional VAE)")
        from src.pipeline import generate_alphabet_neural

        neural_alphabet = generate_alphabet_neural(analysis, checkpoint_path=str(checkpoint_path))
        neural_rec = character_recognition_accuracy(neural_alphabet)
        neural_sim = writer_style_similarity(neural_alphabet, analysis.observed_glyphs)
        logger.info("  Recognition accuracy (template OCR proxy): %.1f%%", neural_rec["__overall__"] * 100)
        if cnn_recognizer is not None:
            neural_rec_cnn = character_recognition_accuracy(neural_alphabet, recognizer=cnn_recognizer)
            logger.info("  Recognition accuracy (trained CNN OCR proxy): %.1f%%", neural_rec_cnn["__overall__"] * 100)
        logger.info("  Writer-style similarity (cosine, rule-based features): %.3f", neural_sim)
        loo_neural = leave_one_out_ssim(image_bgr, label, cfg, use_neural=True)
        if loo_neural:
            logger.info("  Leave-one-out SSIM per char: %s", {k: round(v, 3) for k, v in loo_neural.items()})
            logger.info("  Leave-one-out SSIM mean: %.3f", float(np.mean(list(loo_neural.values()))))
    else:
        logger.info("-" * 70)
        logger.info("Skipping neural backend (no checkpoint at %s)", checkpoint_path)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
