"""Assembles a personalized alphabet from either generator backend into
the on-disk format described in section 13 of the design brief:

    outputs/alphabets/<session_id>/lowercase/a.png ... z.png
    outputs/alphabets/<session_id>/glyph_metadata.json

Both the BASELINE (`BaselineGlyphGenerator`) and the PROPOSED
(`StyleEncoderInference`) backends produce the same in-memory shape
(`dict[str, list[np.ndarray]]`), so this module is agnostic to which one
built the alphabet — swapping backends doesn't change the export format.
"""
from __future__ import annotations

import json
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class AlphabetMetadata:
    generator_backend: str
    observed_chars: list[str]
    generated_chars: list[str]
    created_at: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)


class AlphabetBuilder:
    """Writes a generated alphabet (+ metadata) to disk and reloads it for the renderer."""

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)

    def save(
        self,
        alphabet: dict[str, list[np.ndarray]],
        metadata: AlphabetMetadata,
        session_id: str,
        charset_case: str = "lowercase",
    ) -> Path:
        session_dir = self.output_root / session_id
        case_dir = session_dir / charset_case
        case_dir.mkdir(parents=True, exist_ok=True)

        glyph_meta = {}
        for char, variants in alphabet.items():
            for v_idx, img in enumerate(variants):
                fname = f"{char}.png" if v_idx == 0 else f"{char}_variant{v_idx}.png"
                cv2.imwrite(str(case_dir / fname), img)
            h, w = variants[0].shape[:2]
            glyph_meta[char] = {
                "num_variants": len(variants),
                "width": int(w),
                "height": int(h),
                "observed": char in metadata.observed_chars,
            }

        full_meta = {
            "generator_backend": metadata.generator_backend,
            "observed_chars": metadata.observed_chars,
            "generated_chars": metadata.generated_chars,
            "created_at": metadata.created_at,
            "charset_case": charset_case,
            "glyphs": glyph_meta,
            **metadata.extra,
        }
        with (session_dir / "glyph_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2)

        logger.info("Saved alphabet (%d chars) to %s", len(alphabet), case_dir)
        return session_dir

    def load(self, session_id: str, charset_case: str = "lowercase") -> tuple[dict[str, list[np.ndarray]], dict]:
        session_dir = self.output_root / session_id
        case_dir = session_dir / charset_case
        with (session_dir / "glyph_metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        alphabet: dict[str, list[np.ndarray]] = {}
        for char in string.ascii_lowercase if charset_case == "lowercase" else string.ascii_uppercase:
            if char not in meta["glyphs"]:
                continue
            n_variants = meta["glyphs"][char]["num_variants"]
            imgs = []
            for v_idx in range(n_variants):
                fname = f"{char}.png" if v_idx == 0 else f"{char}_variant{v_idx}.png"
                path = case_dir / fname
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    imgs.append(img)
            if imgs:
                alphabet[char] = imgs
        return alphabet, meta
