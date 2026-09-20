"""Assembles a personalized alphabet from either generator backend into
the on-disk format described in section 13 of the design brief:

    outputs/alphabets/<session_id>/lowercase/a.png ... z.png
    outputs/alphabets/<session_id>/glyph_metadata.json

Both the BASELINE (`BaselineGlyphGenerator`) and the PROPOSED
(`StyleEncoderInference`) backends produce the same in-memory shape
(`dict[str, list[GlyphBitmap]]`), so this module is agnostic to which one
built the alphabet — swapping backends doesn't change the export format.
Glyph PNGs are saved at native resolution; each variant's `GlyphMetrics` goes
into glyph_metadata.json (`glyphs.<c>.metrics`), which is what makes a saved
alphabet renderable (sizes are reconciled by x-height at render time).
"""
from __future__ import annotations

import json
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.generator.glyph import GlyphBitmap, GlyphMetrics
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
        alphabet: dict[str, list[GlyphBitmap]],
        metadata: AlphabetMetadata,
        session_id: str,
        charset_case: str = "lowercase",
    ) -> Path:
        session_dir = self.output_root / session_id
        case_dir = session_dir / charset_case
        case_dir.mkdir(parents=True, exist_ok=True)

        glyph_meta = {}
        for char, variants in alphabet.items():
            for v_idx, glyph in enumerate(variants):
                fname = f"{char}.png" if v_idx == 0 else f"{char}_variant{v_idx}.png"
                cv2.imwrite(str(case_dir / fname), glyph.image)
            h, w = variants[0].image.shape[:2]
            glyph_meta[char] = {
                "num_variants": len(variants),
                "width": int(w),
                "height": int(h),
                "observed": char in metadata.observed_chars,
                "metrics": [g.metrics.to_dict() for g in variants],
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

    def load(self, session_id: str, charset_case: str = "lowercase") -> tuple[dict[str, list[GlyphBitmap]], dict]:
        session_dir = self.output_root / session_id
        case_dir = session_dir / charset_case
        with (session_dir / "glyph_metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        alphabet: dict[str, list[GlyphBitmap]] = {}
        for char in string.ascii_lowercase if charset_case == "lowercase" else string.ascii_uppercase:
            if char not in meta["glyphs"]:
                continue
            entry = meta["glyphs"][char]
            if "metrics" not in entry:
                raise ValueError(
                    f"Alphabet {session_id!r} was saved before glyph metrics existed (no 'metrics' for {char!r}); "
                    "regenerate it with scripts/generate_alphabet.py."
                )
            glyphs = []
            for v_idx in range(entry["num_variants"]):
                fname = f"{char}.png" if v_idx == 0 else f"{char}_variant{v_idx}.png"
                img = cv2.imread(str(case_dir / fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    glyphs.append(GlyphBitmap(img, GlyphMetrics.from_dict(entry["metrics"][v_idx])))
            if glyphs:
                alphabet[char] = glyphs
        return alphabet, meta
