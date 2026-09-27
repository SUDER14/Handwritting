"""Task 4.1: line image -> words by vertical projection profile, with the word-gap threshold LEARNED on train lines.

No word boxes are available (IAM word boxes are not in the Teklia join), so the threshold is learned from what the
train split does have: each line's transcription gives its word count N, i.e. N-1 inter-word gaps. For a candidate
threshold t, a line is segmented at every ink-free column run of width >= t; t is chosen to maximise the fraction of
train lines whose predicted word count equals N exactly.

Everything is measured on the canonical line (Otsu ink, resized to height 64 -- the same crop the writer-ID and CRNN
instruments use), so gap widths are in "64-px-line" units and comparable across scans. Tiny specks (< MIN_SPECK_AREA
pixels at that height) are removed first so dust does not split a gap.

Word definition (fixed before measuring, used for learning and for GATE 4): whitespace tokens of the transcription
that contain a letter or digit; a punctuation-only token ('.', ',', '"') is written attached to its neighbour and is
merged into the previous word (src.generator.vatr_backend.transcription_words).
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_SPECK_AREA = 3
HEIGHT = 64


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


def column_gaps(ink: np.ndarray) -> tuple[list[tuple[int, int, int]], int, int]:
    """Ink-free column runs strictly inside the ink span: [(width, start, end)], plus the span (x0, x1)."""
    cols = (np.asarray(ink) > 127).any(axis=0)
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


def predicted_word_count(ink: np.ndarray, threshold: float) -> int:
    gaps, x0, x1 = column_gaps(ink)
    if x1 <= x0:
        return 0
    return 1 + sum(1 for w, _, _ in gaps if w >= threshold)


def learn_gap_threshold(lines: list[tuple[np.ndarray, int]], candidates=None) -> dict:
    """lines = [(canonical ink, true word count)] -> {threshold, exact_acc, within1_acc, curve}. Ties -> smallest t."""
    gap_lists = []
    for ink, n in lines:
        gaps, x0, x1 = column_gaps(ink)
        gap_lists.append((np.array([w for w, _, _ in gaps], float), n, x1 > x0))
    candidates = list(candidates if candidates is not None else range(1, 61))
    curve = []
    for t in candidates:
        pred = np.array([(1 + int((g >= t).sum())) if has_ink else 0 for g, _, has_ink in gap_lists])
        true = np.array([n for _, n, _ in gap_lists])
        curve.append((t, float(np.mean(pred == true)), float(np.mean(np.abs(pred - true) <= 1))))
    best = max(curve, key=lambda c: (c[1], -c[0]))
    return {"threshold": best[0], "exact_acc": best[1], "within1_acc": best[2], "n_lines": len(lines),
            "curve": [{"t": t, "exact": e, "within1": w} for t, e, w in curve]}


def segment_words(gray: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Line (native resolution, dark ink on light) -> word crops at native resolution, each trimmed to its ink box and
    returned dark-on-light, left to right. Cuts are the centres of canonical gaps >= threshold, mapped back to native x."""
    g = np.asarray(gray, np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    ink = canonical(g)
    gaps, x0, x1 = column_gaps(ink)
    if x1 <= x0:
        return []
    sx = g.shape[1] / ink.shape[1]
    cuts = [int(round((s + e) / 2 * sx)) for w, s, e in gaps if w >= threshold]
    _, native_ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bounds = [0, *cuts, g.shape[1]]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = native_ink[:, a:b] > 0
        ys, xs = np.nonzero(m)
        if len(xs) < MIN_SPECK_AREA:
            continue
        out.append(g[ys.min(): ys.max() + 1, a + xs.min(): a + xs.max() + 1])
    return out


def normalise_height(crop: np.ndarray, height: int = 32) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)


def references_from_lines(line_grays: list[np.ndarray], threshold: float, height: int = 32) -> list[np.ndarray]:
    """4.1 user-sample path: line images (no transcription) -> height-normalised word crops for the generator."""
    return [normalise_height(c, height) for g in line_grays for c in segment_words(g, threshold)]
