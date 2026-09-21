"""Calibration of the CNN legibility scorer on REAL handwriting.

The legibility metric reads generated text back with segmentation + `CNNRecognizer`. Before any generation
result can be interpreted we need its FLOOR: how well does the same scorer read genuine human handwriting
of known text? If real handwriting already scores a high CER, the scorer cannot resolve differences between
generators (they would all be lost in the same noise).

The scoring path here is the one evaluate.py uses on generated images (segment the binary line, classify each
glyph, compare letters to the reference with edit distance), preceded by the normal preprocessing because real
crops are photographs/scans rather than renders. Reference text = the transcription reduced to lower-case
letters (the CNN is case-agnostic a-z; spaces and punctuation are dropped on both sides).

Two null decoders are reported so the number can be read against chance: uniformly random letters (one per
detected glyph) and a constant letter. A scorer that barely beats these is not measuring anything.
"""
from __future__ import annotations

import hashlib
import string

import cv2
import numpy as np

from src.evaluation.harness import letters_only
from src.evaluation.text_metrics import cer, levenshtein
from src.preprocessing.pipeline import Preprocessor
from src.segmentation.segmenter import Segmenter


def select_crops(dataset, n: int, seed: int, min_letters: int) -> list[int]:
    """Deterministic hash-ranked sample of usable crops (>= min_letters letters)."""
    eligible = [i for i, s in enumerate(dataset.samples) if len(letters_only(s.transcription)) >= min_letters]
    eligible.sort(key=lambda i: hashlib.sha256(f"{seed}:calib:{dataset.samples[i].sample_id}".encode()).hexdigest())
    return eligible[:n]


def score_crop(gray: np.ndarray, reference: str, recognizer, cfg: dict) -> dict:
    binary = Preprocessor(cfg).process(image_bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), n_chars=len(reference)).cropped
    glyphs = Segmenter(cfg).segment(binary).glyphs
    decoded = "".join(recognizer.recognize([g.image for g in glyphs])) if glyphs else ""
    return {"reference": reference, "decoded": decoded, "n_glyphs": len(glyphs), "n_ref": len(reference),
            "edit_distance": levenshtein(reference, decoded), "cer": cer(reference, decoded)}


def run_calibration(dataset, recognizer, cfg: dict, n_crops: int, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    min_letters = cfg["evaluation"]["min_reference_letters"]
    idx = select_crops(dataset, n_crops, seed, min_letters)
    records = []
    for i in idx:
        s = dataset.samples[i]
        ref = letters_only(s.transcription)
        try:
            rec = score_crop(dataset.read_native(i), ref, recognizer, cfg)
        except Exception as e:  # noqa: BLE001 -- recorded and counted, never hidden
            rec = {"reference": ref, "error": f"{type(e).__name__}: {e}", "n_ref": len(ref)}
        rec.update({"sample_id": s.sample_id, "writer_id": s.writer_id, "transcription": s.transcription})
        records.append(rec)

    ok = [r for r in records if "error" not in r]
    if not ok:
        raise RuntimeError(f"no crop could be scored; first error: {records[0].get('error') if records else 'no crops'}")
    for r in ok:                                   # null decoders: same glyph count as the scorer found
        rand = "".join(rng.choice(list(string.ascii_lowercase), size=r["n_glyphs"])) if r["n_glyphs"] else ""
        r["cer_random_letters"] = cer(r["reference"], rand)
        r["cer_constant_e"] = cer(r["reference"], "e" * r["n_glyphs"])
    cers = np.array([r["cer"] for r in ok])
    total_ref = sum(r["n_ref"] for r in ok)
    agg = {
        "n_requested": n_crops, "n_scored": len(ok), "n_failed": len(records) - len(ok),
        "n_writers": len({r["writer_id"] for r in ok}), "n_reference_letters": total_ref,
        "cer_micro": sum(r["edit_distance"] for r in ok) / total_ref,
        "cer_mean": float(cers.mean()), "cer_median": float(np.median(cers)),
        "cer_p25": float(np.percentile(cers, 25)), "cer_p75": float(np.percentile(cers, 75)),
        "cer_p90": float(np.percentile(cers, 90)),
        "glyph_count_match_rate": float(np.mean([r["n_glyphs"] == r["n_ref"] for r in ok])),
        "glyphs_per_reference_letter": float(sum(r["n_glyphs"] for r in ok) / total_ref),
        "null_cer_random_letters": float(np.mean([r["cer_random_letters"] for r in ok])),
        "null_cer_constant_e": float(np.mean([r["cer_constant_e"] for r in ok])),
    }
    return records, agg
