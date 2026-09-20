"""CLI: generate a personalized alphabet from a handwriting sample image.

Usage:
    python scripts/generate_alphabet.py --image data/samples/quick_sample.png --label quick --backend baseline
    python scripts/generate_alphabet.py --image data/samples/quick_sample.png --label quick --backend neural
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.generator.alphabet_builder import AlphabetBuilder, AlphabetMetadata
from src.pipeline import analyze_sample, generate_alphabet_baseline, generate_alphabet_neural
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the handwriting sample image.")
    parser.add_argument("--label", required=True, help="Ground-truth word written in the image (letters only).")
    parser.add_argument("--backend", choices=["baseline", "neural"], default="baseline")
    parser.add_argument("--session-id", default="cli_session")
    parser.add_argument("--checkpoint", default=None, help="Override the trained VAE checkpoint path.")
    args = parser.parse_args()

    cfg = load_config()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    label = "".join(c for c in args.label.lower() if c.isalpha())
    analysis = analyze_sample(image_bgr, label, cfg)
    logger.info("Detected characters: %s", analysis.chars)

    if args.backend == "baseline":
        alphabet = generate_alphabet_baseline(analysis, cfg)
    else:
        alphabet = generate_alphabet_neural(analysis, checkpoint_path=args.checkpoint)

    builder = AlphabetBuilder(resolve_path("outputs/alphabets"))
    meta = AlphabetMetadata(
        generator_backend=args.backend,
        observed_chars=list(analysis.observed_glyphs.keys()),
        generated_chars=[c for c in "abcdefghijklmnopqrstuvwxyz" if c not in analysis.observed_glyphs],
    )
    out_dir = builder.save(alphabet, meta, args.session_id)
    logger.info("Alphabet written to %s", out_dir)


if __name__ == "__main__":
    main()
