"""Task 4.1: line image -> words by vertical projection profile, with the word-gap rule LEARNED on train lines.

No word boxes are available (IAM word boxes are not in the Teklia join), so the rule is learned from what the train
split does have: each line's transcription gives its word count N, i.e. N-1 inter-word gaps.

Rule (chosen on VAL among global / per-line-relative variants, parameters fitted on TRAIN; see ROADMAP_STATE 4.1):
  1. deslant the line: the shear in [-45, 45] deg (2.5 deg steps) that makes the vertical projection sharpest, so
     slanted neighbouring words stop overlapping in columns;
  2. a column is "empty" if it holds <= COL_TOL ink pixels (a descender tail or t-bar reaching under the next word);
  3. cut at every empty-column run of width >= max(floor, rel_max * the line's widest gap): a per-line threshold,
     because some writers space words as tightly as others space letters (one global threshold reached only 0.76
     within +/-1 on val).
(floor, rel_max) maximise the fraction of train lines with |predicted - N| <= 1 (learn_gap_rule).

Everything is measured on the canonical line (Otsu ink, resized to height 64 -- the crop the writer-ID and CRNN
instruments use), so widths are in "64-px-line" units. Tiny specks (< MIN_SPECK_AREA px) are removed first.

Word definition (fixed before measuring; used for learning and for GATE 4): whitespace tokens of the transcription
that contain a letter or digit; a punctuation-only token ('.', ',', '"') is merged into the previous word
(src.generator.vatr_backend.transcription_words).
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_SPECK_AREA = 3
HEIGHT = 64
COL_TOL = 2
DESLANT_RANGE = np.arange(-45.0, 45.1, 2.5)
DEFAULT_RULE = {"floor": 11.0, "rel_max": 0.4}    # the learned values live in config segmentation.word_gap


def canonical(gray: np.ndarray, height: int = HEIGHT) -> np.ndarray:
    """Dark-ink-on-light line -> ink-high uint8 (0/255) at `height` px, specks removed."""
    from src.writer_id.data import canonical_from_iam_gray
    g = np.asarray(gray, np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    return despeckle(canonical_from_iam_gray(g, height))


def despeckle(ink: np.ndarray) -> np.ndarray:
    b = (np.asarray(ink) > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_SPECK_AREA
    return (keep[lab] * 255).astype(np.uint8)


def shear(ink: np.ndarray, deg: float) -> tuple[np.ndarray, int]:
    """x' = x - tan(deg) * (y - h) + pad: the bottom row is only shifted by `pad`. Returns (sheared, pad)."""
    h = ink.shape[0]
    s = -np.tan(np.radians(deg))
    pad = int(abs(s) * h) + 2
    p = cv2.copyMakeBorder(np.asarray(ink), 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    m = np.array([[1, s, -s * h], [0, 1, 0]], np.float32)
    return cv2.warpAffine(p, m, (p.shape[1], p.shape[0]), flags=cv2.INTER_NEAREST), pad


def slant_angle(ink: np.ndarray) -> float:
    """The CORRECTING shear (deg) for shear(): the angle whose vertical ink projection is sharpest (sum of squared
    column counts). Writing sheared by +a reads back as -a; deslant() applies it."""
    ys, xs = np.nonzero(np.asarray(ink) > 127)
    if len(xs) < 10:
        return 0.0
    h = ink.shape[0]
    best, best_deg = -1.0, 0.0
    for deg in DESLANT_RANGE:
        x = np.round(xs - np.tan(np.radians(deg)) * (ys - h)).astype(np.int64)
        score = float((np.bincount(x - x.min()).astype(np.float64) ** 2).sum())
        if score > best:
            best, best_deg = score, float(deg)
    return best_deg


def deslant(ink: np.ndarray) -> np.ndarray:
    deg = slant_angle(ink)
    return shear(ink, deg)[0] if deg else np.asarray(ink)


def column_gaps(ink: np.ndarray, col_tol: int = 0) -> tuple[list[tuple[int, int, int]], int, int]:
    """Empty-column runs (<= col_tol ink px) strictly inside the ink span: [(width, start, end)], and (x0, x1)."""
    cols = (np.asarray(ink) > 127).sum(axis=0) > col_tol
    xs = np.nonzero(cols)[0]
    if len(xs) == 0:
        return [], 0, 0
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    gaps, x = [], x0
    while x < x1:
        if not cols[x]:
            s = x
            while x < x1 and not cols[x]:
                x += 1
            gaps.append((x - s, s, x))
        else:
            x += 1
    return gaps, x0, x1


def line_threshold(gap_widths, rule: dict) -> float:
    g = np.asarray(gap_widths, float)
    return max(rule["floor"], rule["rel_max"] * g.max()) if len(g) else rule["floor"]


def line_gaps(ink: np.ndarray):
    """(deslanted ink, tolerant column gaps, x0, x1)."""
    d = deslant(ink)
    gaps, x0, x1 = column_gaps(d, COL_TOL)
    return d, gaps, x0, x1


def predicted_word_count(ink: np.ndarray, rule: dict = DEFAULT_RULE) -> int:
    _, gaps, x0, x1 = line_gaps(ink)
    if x1 <= x0:
        return 0
    t = line_threshold([w for w, _, _ in gaps], rule)
    return 1 + sum(1 for w, _, _ in gaps if w >= t)


def learn_gap_rule(lines: list[tuple[np.ndarray, int]], floors=range(4, 20),
                   rel_maxes=np.arange(0.2, 1.001, 0.05)) -> dict:
    """lines = [(canonical ink, true word count)] -> {rule: {floor, rel_max}, within1_acc, exact_acc, ...}, maximising
    within-1 accuracy (ties: exact accuracy, then the smaller floor / rel_max)."""
    gl = []
    for ink, n in lines:
        _, gaps, x0, x1 = line_gaps(ink)
        gl.append((np.array([w for w, _, _ in gaps], float), x1 > x0))
    true = np.array([n for _, n in lines])
    best = None
    for a in floors:
        for b in rel_maxes:
            rule = {"floor": float(a), "rel_max": round(float(b), 4)}
            pred = np.array([(1 + int((g >= line_threshold(g, rule)).sum())) if ok else 0 for g, ok in gl])
            key = (float(np.mean(np.abs(pred - true) <= 1)), float(np.mean(pred == true)), -float(a), -float(b))
            if best is None or key > best[0]:
                best = (key, rule)
    (w1, ex, _, _), rule = best
    return {"rule": rule, "within1_acc": w1, "exact_acc": ex, "n_lines": len(lines), "col_tol": COL_TOL,
            "deslant_range_deg": [float(DESLANT_RANGE[0]), float(DESLANT_RANGE[-1])]}


def segment_words(gray: np.ndarray, rule: dict = DEFAULT_RULE) -> list[np.ndarray]:
    """Line (native resolution, dark ink on light) -> word crops at native resolution, left to right, each trimmed to
    its ink box, dark-on-light. Cuts are found in the deslanted canonical frame; the native line is deslanted by the
    same angle, cut there, and each piece is sheared back upright before trimming."""
    g = np.asarray(gray, np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    ink = canonical(g)
    deg = slant_angle(ink)
    d, pad = shear(ink, deg) if deg else (ink, 0)
    gaps, x0, x1 = column_gaps(d, COL_TOL)
    if x1 <= x0:
        return []
    t = line_threshold([w for w, _, _ in gaps], rule)
    _, native_ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    s = g.shape[0] / ink.shape[0]                                 # canonical -> native scale (height-driven resize)
    nd, npad = shear(native_ink, deg) if deg else (native_ink, 0)
    cuts = [int(round(((a + b) / 2 - pad) * s + npad)) for w, a, b in gaps if w >= t]
    bounds = [0, *cuts, nd.shape[1]]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        piece = np.zeros_like(nd)
        piece[:, a:b] = nd[:, a:b]
        up = shear(piece, -deg)[0] if deg else piece
        ys, xs = np.nonzero(up > 0)
        if len(xs) < MIN_SPECK_AREA:
            continue
        out.append((255 - up[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]).astype(np.uint8))
    return out


def normalise_height(crop: np.ndarray, height: int = 32) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def references_from_lines(line_grays: list[np.ndarray], rule: dict = DEFAULT_RULE, height: int = 32) -> list[np.ndarray]:
    """4.1 user-sample path: line images (no transcription) -> height-normalised word crops for the generator."""
    return [normalise_height(c, height) for g in line_grays for c in segment_words(g, rule)]
