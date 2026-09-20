"""Creates a synthetic handwriting-like sample image for pipeline testing.

We don't have a real scanned handwriting sample in this environment, so we
render a word with a handwriting-style system font (Segoe Print — visually
distinct from Segoe Script, which the baseline generator uses as its
reference font, so this remains a meaningful, non-circular test) and add
mild photo-realistic noise/rotation/lighting variation. This is ONLY a
development/testing aid (smoke tests only, never an evaluation set — real
evaluation uses writer-disjoint IAM splits, see src/data/iam.py) — real usage
is via actual uploaded photos in the Streamlit app.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.config import resolve_path

FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoepr.ttf",
    "C:/Windows/Fonts/comic.ttf",
]


def make_sample(word: str, out_path: Path, font_size: int = 90, noisy: bool = True) -> None:
    font = None
    for path in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    canvas_w, canvas_h = 900, 260
    img = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_w - w) / 2 - bbox[0]
    y = (canvas_h - h) / 2 - bbox[1]
    draw.text((x, y), word, fill=0, font=font)

    arr = np.array(img)

    if noisy:
        # Slight rotation to simulate a not-perfectly-aligned photo.
        rot_M = cv2.getRotationMatrix2D((canvas_w / 2, canvas_h / 2), 3.5, 1.0)
        arr = cv2.warpAffine(arr, rot_M, (canvas_w, canvas_h), borderValue=255)
        # Gaussian noise.
        noise = np.random.default_rng(1).normal(0, 8, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Slight uneven lighting.
        gradient = np.tile(np.linspace(-15, 15, canvas_w), (canvas_h, 1))
        arr = np.clip(arr.astype(np.float32) + gradient, 0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))
    print(f"Wrote synthetic sample to {out_path}")


if __name__ == "__main__":
    make_sample("quick", resolve_path("data/samples/quick_sample.png"))
    make_sample("hello", resolve_path("data/samples/hello_sample.png"))
