"""Quantitative evaluation (section 20 of the design brief).

Three metrics are implemented, chosen for being computable WITHOUT
extra manual data collection (important — human evaluation, the fourth
metric in the brief, is documented as a protocol in docs/experiments.md
but requires human raters and isn't automatable here):

1. `ssim_against_reference` — structural similarity between a generated
   glyph and a real reference glyph, when a ground-truth image for that
   exact character exists (e.g. leave-one-out: hide "q" from the sample,
   generate it, compare to the real "q").
2. `character_recognition_accuracy` — runs the non-learned template
   recognizer (src/recognition/template_matching.py) over the generated
   alphabet as an OCR proxy: "does this generated glyph still look
   readable as the intended letter?"
3. `writer_style_similarity` — cosine similarity between the rule-based
   StyleProfile feature vectors (src/features/style_extractor.py) of the
   generated alphabet and the original sample. This is a PROXY for
   "writer identification" (the brief's stated ideal) — training an
   actual writer-verification model needs a writer-labeled dataset
   (e.g. IAM with writer IDs) which is out of scope for the MVP; see
   docs/limitations.md.
"""
from __future__ import annotations

import string

import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.features.style_extractor import StyleFeatureExtractor
from src.recognition.base import Recognizer
from src.recognition.template_matching import TemplateMatchingRecognizer


def ssim_against_reference(generated: np.ndarray, reference: np.ndarray) -> float:
    """SSIM between a generated glyph and a real reference glyph of the same character.

    Both images are resized to a common size before comparison.
    """
    import cv2
    size = (64, 64)
    g = cv2.resize(generated, size, interpolation=cv2.INTER_AREA)
    r = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    score, _ = ssim(g, r, full=True, data_range=255)
    return float(score)


def character_recognition_accuracy(
    alphabet: dict,
    charset: str = string.ascii_lowercase,
    recognizer: Recognizer | None = None,
) -> dict[str, float]:
    """Fraction of generated glyphs an OCR-proxy recognizer classifies correctly.

    Defaults to the non-learned template-matching recognizer (as before).
    Pass a trained `CNNRecognizer` (src/recognition/cnn_recognizer.py) as
    `recognizer` for a much more meaningful "is this glyph actually
    readable" proxy — see `scripts/evaluate.py`, which reports both when a
    CNN checkpoint is available.

    Returns both per-character correctness (1.0/0.0) and the overall accuracy
    under the key "__overall__".
    """
    if recognizer is None:
        recognizer = TemplateMatchingRecognizer(charset=charset)
    results = {}
    correct = 0
    total = 0
    for char in charset:
        if char not in alphabet or not alphabet[char]:
            continue
        img = alphabet[char][0].image   # GlyphBitmap -> pixels
        predicted = recognizer.recognize([img])[0]
        is_correct = float(predicted == char)
        results[char] = is_correct
        correct += is_correct
        total += 1
    results["__overall__"] = correct / total if total else 0.0
    return results


def writer_style_similarity(
    generated_alphabet: dict,
    reference_glyphs: dict[str, np.ndarray],
) -> float:
    """Cosine similarity between the writer-level rule-based feature vectors
    of the generated alphabet and the original reference sample.

    A proxy for writer-identity preservation (see module docstring).
    """
    extractor = StyleFeatureExtractor()

    def profile_vector(glyphs: dict[str, np.ndarray]) -> np.ndarray:
        feats = [extractor.extract_glyph_features(c, g) for c, g in glyphs.items()]
        if not feats:
            return np.zeros(6, dtype=np.float32)
        return np.array([
            np.mean([f.aspect_ratio for f in feats]),
            np.mean([f.slant_deg for f in feats]),
            np.mean([f.stroke_width_px for f in feats]),
            np.mean([f.ink_density for f in feats]),
            np.mean([f.curvature_score for f in feats]),
            np.std([f.slant_deg for f in feats]),
        ], dtype=np.float32)

    gen_glyphs = {c: v[0].image for c, v in generated_alphabet.items() if v}
    v1 = profile_vector(gen_glyphs)
    v2 = profile_vector(reference_glyphs)

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(np.dot(v1, v2) / denom)
