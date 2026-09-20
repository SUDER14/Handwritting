"""CLI: render arbitrary text using a previously generated alphabet.

Usage:
    python scripts/generate_alphabet.py --image data/samples/quick_sample.png --label quick --session-id demo
    python scripts/generate_text.py --session-id demo --text "the quick fox" --out outputs/generated_text/demo.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.generator.alphabet_builder import AlphabetBuilder
from src.renderer.text_renderer import TextRenderer
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True, help="Session ID used by generate_alphabet.py.")
    parser.add_argument("--text", required=True, help="Text to render in the personalized handwriting style.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    args = parser.parse_args()

    cfg = load_config()
    builder = AlphabetBuilder(resolve_path("outputs/alphabets"))
    alphabet, meta = builder.load(args.session_id)
    logger.info("Loaded alphabet (backend=%s) for session %s", meta["generator_backend"], args.session_id)

    renderer = TextRenderer(alphabet, cfg)
    rendered = renderer.render_to_bgr_on_white(args.text.lower())

    out_path = Path(args.out) if args.out else resolve_path("outputs/generated_text") / f"{args.session_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rendered)
    logger.info("Wrote rendered handwriting to %s", out_path)


if __name__ == "__main__":
    main()
