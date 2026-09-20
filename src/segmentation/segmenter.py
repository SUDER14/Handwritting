"""Character segmentation for handwritten word/line images.

Splits a binarized, deskewed word image (ink=255 on black background) into
individual glyph candidates.

Connected-component segmentation assumes each character is (mostly) a
separate ink blob. It handles printed/disconnected handwriting well and
correctly reunites multi-part characters (the dot on "i"/"j", crossed "t")
via a proximity-merge step. Left on its own it would UNDER-segment true
cursive script, where adjacent letters share a continuous stroke (e.g. a
joined "cl") and show up as one merged component — so a second pass,
`_split_touching_glyphs`, looks for components much wider than an expected
single-character width and splits them at valleys in the vertical
ink-density (projection) profile: even where cursive strokes never fully
separate into disjoint components, the ink still thins out where one
letter's stroke hands off to the next, and that's exactly what a valley in
the column-sum profile captures. This is a heuristic, not a learned
segmentation model — genuinely ambiguous or heavily overlapping cursive can
still be misattributed (see docs/limitations.md) — but it materially
improves on treating every touching pair as unsplittable. The whole
segmentation strategy sits behind the `Segmenter` interface so it can be
replaced (e.g. with a learned segmentation network) without touching any
other module.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Glyph:
    """One segmented character candidate."""

    index: int              # left-to-right order in the line
    bbox: tuple[int, int, int, int]   # x, y, w, h in the source image
    image: np.ndarray       # cropped binary mask (ink=255), tight to bbox
    centroid: tuple[float, float]


@dataclass
class SegmentationResult:
    glyphs: list[Glyph]
    baseline_y: int
    x_height_top_y: int      # approximate top of lowercase x-height band
    line_image: np.ndarray


class Segmenter:
    """Connected-component based character segmenter with dot/diacritic merging."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        self.cfg = cfg["segmentation"]

    def _connected_components(self, binary: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        """Return list of (x, y, w, h, area) for each foreground component."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        boxes = []
        min_area = self.cfg["min_component_area"]
        for label in range(1, num_labels):  # skip background label 0
            x, y, w, h, area = stats[label]
            if area < min_area:
                continue
            boxes.append((int(x), int(y), int(w), int(h), int(area)))
        return boxes

    def _merge_diacritics(
        self, boxes: list[tuple[int, int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge small components (dots, accents) into the nearest body component below them.

        A component is treated as a diacritic candidate if it is small and
        sits above a taller component with which it horizontally overlaps.
        """
        if not boxes:
            return []

        # Heuristic: "body" components are the tallest ones; anything much
        # shorter that overlaps horizontally with a body component and sits
        # above it is merged into that body.
        heights = [h for _, _, _, h, _ in boxes]
        median_h = float(np.median(heights))

        bodies = [b for b in boxes if b[3] >= 0.5 * median_h]
        small = [b for b in boxes if b[3] < 0.5 * median_h]

        merged = {i: list(b[:4]) for i, b in enumerate(bodies)}
        gap = self.cfg["merge_gap_px"]

        for sx, sy, sw, sh, _ in small:
            best_i, best_overlap = None, 0
            for i, (bx, by, bw, bh, _area) in enumerate(bodies):
                overlap = max(0, min(sx + sw, bx + bw) - max(sx, bx))
                vertical_gap = by - (sy + sh)
                if overlap > 0 and -gap <= vertical_gap <= 4 * gap:
                    if overlap > best_overlap:
                        best_overlap, best_i = overlap, i
            if best_i is not None:
                bx, by, bw, bh = merged[best_i]
                nx0, ny0 = min(bx, sx), min(by, sy)
                nx1, ny1 = max(bx + bw, sx + sw), max(by + bh, sy + sh)
                merged[best_i] = [nx0, ny0, nx1 - nx0, ny1 - ny0]
            else:
                # No body nearby (rare) — keep it as its own glyph.
                merged[len(merged)] = [sx, sy, sw, sh]

        return [tuple(v) for v in merged.values()]

    def _projection_profile_splits(self, crop: np.ndarray, expected_w: float, min_w: int) -> list[int]:
        """Column indices (local x within `crop`) at which to cut a touching-glyph blob.

        Classic cursive word-splitting heuristic: estimate how many characters
        the blob probably contains from its width vs. `expected_w`, then pick
        that many local minima of the smoothed vertical ink-density (column
        sum) profile, spaced apart, as cut points.
        """
        col_sums = (crop > 0).sum(axis=0).astype(np.float32)
        if col_sums.max() == 0:
            return []
        n_chars = max(1, int(round(crop.shape[1] / expected_w)))
        if n_chars < 2:
            return []
        kernel = np.ones(3, dtype=np.float32) / 3.0
        smoothed = np.convolve(col_sums, kernel, mode="same")

        target_cuts = n_chars - 1
        margin = max(min_w, int(expected_w * 0.3), 1)
        width = len(smoothed)
        search = smoothed.copy()
        search[:min(margin, width)] = np.inf
        search[max(0, width - margin):] = np.inf

        cuts: list[int] = []
        for _ in range(target_cuts):
            if np.all(np.isinf(search)):
                break
            idx = int(np.argmin(search))
            if np.isinf(search[idx]):
                break
            cuts.append(idx)
            lo, hi = max(0, idx - margin), min(len(search), idx + margin)
            search[lo:hi] = np.inf
        return sorted(cuts)

    def _split_touching_glyphs(
        self, boxes: list[tuple[int, int, int, int]], binary: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Split components far wider than an expected single-character width.

        The expected width is the median width of "normal-looking" boxes
        (not already suspiciously wide relative to height); if every box in
        the line looks wide (e.g. an entire cursive word is one connected
        component), fall back to a height-derived estimate instead.
        """
        if not self.cfg.get("split_touching_glyphs", True) or not boxes:
            return boxes

        heights = [h for _, _, _, h in boxes]
        median_h = float(np.median(heights)) if heights else 0.0
        normal_widths = [w for _, _, w, h in boxes if median_h > 0 and w <= 1.3 * median_h]
        if normal_widths:
            expected_w = float(np.median(normal_widths))
        else:
            expected_w = 0.75 * median_h
        if expected_w <= 0:
            return boxes

        ratio = self.cfg.get("wide_component_width_ratio", 1.6)
        min_w = self.cfg.get("min_split_segment_px", 6)

        result = []
        for (x, y, w, h) in boxes:
            if w > ratio * expected_w and w > 2 * min_w:
                crop = binary[y:y + h, x:x + w]
                splits = self._projection_profile_splits(crop, expected_w, min_w)
                if splits:
                    prev = 0
                    for cut in splits + [w]:
                        seg_w = cut - prev
                        if seg_w >= min_w:
                            result.append((x + prev, y, seg_w, h))
                        prev = cut
                    continue
            result.append((x, y, w, h))
        return result

    def segment(self, binary: np.ndarray) -> SegmentationResult:
        """Segment a binarized line image into ordered glyph candidates."""
        boxes_with_area = self._connected_components(binary)
        merged_boxes = self._merge_diacritics(boxes_with_area)
        merged_boxes = self._split_touching_glyphs(merged_boxes, binary)

        # Order left-to-right by x position.
        merged_boxes.sort(key=lambda b: b[0])

        glyphs = []
        bottoms = []
        for i, (x, y, w, h) in enumerate(merged_boxes):
            crop = binary[y:y + h, x:x + w]
            cx, cy = x + w / 2.0, y + h / 2.0
            glyphs.append(Glyph(index=i, bbox=(x, y, w, h), image=crop, centroid=(cx, cy)))
            bottoms.append(y + h)

        baseline_y = int(np.median(bottoms)) if bottoms else binary.shape[0]
        tops = [g.bbox[1] for g in glyphs]
        x_height_top = int(np.median(tops)) if tops else 0

        logger.info("Segmented %d glyph(s) from line of shape %s", len(glyphs), binary.shape)
        return SegmentationResult(
            glyphs=glyphs, baseline_y=baseline_y, x_height_top_y=x_height_top, line_image=binary
        )

    def segment_and_align_to_label(self, binary: np.ndarray, label_text: str) -> SegmentationResult:
        """Segment then reconcile the glyph count against a known ground-truth label.

        This is the practical MVP recognition strategy (config: recognition.backend
        == "labeled"): rather than solving unconstrained handwritten character
        recognition, the user supplies the word they wrote (e.g. "quick"). If
        segmentation finds a different number of components than len(label_text)
        (common with touching cursive letters or split strokes), we deterministically
        reconcile by merging extras (smallest gaps first) or splitting the widest
        glyph evenly, so every glyph still gets a best-effort character label. This
        keeps the pipeline usable while the true segmentation-free step (a learned
        recognizer, see src/recognition/) is a pluggable upgrade.
        """
        result = self.segment(binary)
        n_target = len(label_text)
        glyphs = result.glyphs

        if len(glyphs) > n_target:
            glyphs = self._merge_smallest_gaps(glyphs, n_target)
        elif len(glyphs) < n_target and len(glyphs) > 0:
            glyphs = self._split_widest(glyphs, n_target)

        for i, g in enumerate(glyphs):
            g.index = i

        result.glyphs = glyphs
        return result

    def _merge_smallest_gaps(self, glyphs: list[Glyph], n_target: int) -> list[Glyph]:
        glyphs = sorted(glyphs, key=lambda g: g.bbox[0])
        while len(glyphs) > n_target:
            gaps = [
                glyphs[i + 1].bbox[0] - (glyphs[i].bbox[0] + glyphs[i].bbox[2])
                for i in range(len(glyphs) - 1)
            ]
            merge_at = int(np.argmin(gaps))
            a, b = glyphs[merge_at], glyphs[merge_at + 1]
            ax, ay, aw, ah = a.bbox
            bx, by, bw, bh = b.bbox
            nx0, ny0 = min(ax, bx), min(ay, by)
            nx1, ny1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
            merged_bbox = (nx0, ny0, nx1 - nx0, ny1 - ny0)
            full = a.image if a.image.shape[0] * a.image.shape[1] >= b.image.shape[0] * b.image.shape[1] else b.image
            new_glyph = Glyph(index=0, bbox=merged_bbox, image=full, centroid=((nx0 + nx1) / 2, (ny0 + ny1) / 2))
            glyphs = glyphs[:merge_at] + [new_glyph] + glyphs[merge_at + 2:]
        return glyphs

    def _split_widest(self, glyphs: list[Glyph], n_target: int) -> list[Glyph]:
        glyphs = sorted(glyphs, key=lambda g: g.bbox[0])
        while len(glyphs) < n_target:
            widths = [g.bbox[2] for g in glyphs]
            split_at = int(np.argmax(widths))
            g = glyphs[split_at]
            x, y, w, h = g.bbox
            half_w = w // 2
            left_img = g.image[:, :half_w]
            right_img = g.image[:, half_w:]
            left = Glyph(index=0, bbox=(x, y, half_w, h), image=left_img, centroid=(x + half_w / 2, y + h / 2))
            right = Glyph(
                index=0, bbox=(x + half_w, y, w - half_w, h), image=right_img,
                centroid=(x + half_w + (w - half_w) / 2, y + h / 2),
            )
            glyphs = glyphs[:split_at] + [left, right] + glyphs[split_at + 1:]
        return glyphs
