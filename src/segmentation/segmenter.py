"""Character segmentation for handwritten word/line images.

Splits a binarized, deskewed word image (ink=255 on black background) into
individual glyph candidates.

Connected-component segmentation assumes each character is (mostly) a
separate ink blob. It handles printed/disconnected handwriting well and
correctly reunites multi-part characters (the dot on "i"/"j", crossed "t")
via a stacked-component merge (`_merge_stacked`). Left on its own it would UNDER-segment true
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
    """Connected-component based character segmenter with stacked-component (dot/diacritic) merging."""

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

    def _merge_stacked(
        self, boxes: list[tuple[int, int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge connected components that are vertically stacked with substantial x-overlap.

        This is what puts the dot of an 'i' or 'j' (and the parts of ':', ';', '!', accented
        letters) back onto its stem. Two components A and B are merged when
          * their x-ranges overlap by at least `stack_min_x_overlap` of the NARROWER box
            (0.25 = at least a quarter of the narrower part must sit over the other);
          * they are stacked, not side by side: the vertical overlap is at most `merge_gap_px`
            (a small tolerance for touching/overlapping bboxes) and the vertical gap is at most
            `stack_max_gap_ratio` of the TALLER box's height.
        Merging is transitive (union-find), so dot + stem + descender collapse into one glyph.
        Neighbouring letters are not stacked -- their vertical ranges overlap heavily -- so they
        stay separate even when slant makes their bounding boxes overlap in x.

        Note the size of the dot is deliberately irrelevant: the previous rule only merged
        components under half the median height, and its pairing window needed the dot to sit
        directly over the stem, which an ill-judged deskew easily broke.
        """
        if not boxes:
            return []
        min_x_overlap = self.cfg.get("stack_min_x_overlap", 0.25)
        max_gap_ratio = self.cfg.get("stack_max_gap_ratio", 0.6)
        tolerated_v_overlap = self.cfg["merge_gap_px"]

        n = len(boxes)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            ax, ay, aw, ah, _ = boxes[i]
            for j in range(i + 1, n):
                bx, by, bw, bh, _ = boxes[j]
                x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
                if x_overlap <= 0 or x_overlap < min_x_overlap * min(aw, bw):
                    continue
                v_gap = max(ay, by) - min(ay + ah, by + bh)   # > 0 separated, < 0 overlapping
                if -tolerated_v_overlap <= v_gap <= max_gap_ratio * max(ah, bh):
                    parent[find(i)] = find(j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        merged = []
        for members in groups.values():
            x0 = min(boxes[k][0] for k in members)
            y0 = min(boxes[k][1] for k in members)
            x1 = max(boxes[k][0] + boxes[k][2] for k in members)
            y1 = max(boxes[k][1] + boxes[k][3] for k in members)
            merged.append((x0, y0, x1 - x0, y1 - y0))
        return merged

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
        merged_boxes = self._merge_stacked(boxes_with_area)
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
            glyphs = self._merge_smallest_gaps(glyphs, n_target, binary)
        elif len(glyphs) < n_target and len(glyphs) > 0:
            glyphs = self._split_widest(glyphs, n_target)

        for i, g in enumerate(glyphs):
            g.index = i

        result.glyphs = glyphs
        return result

    def _merge_smallest_gaps(self, glyphs: list[Glyph], n_target: int, binary: np.ndarray) -> list[Glyph]:
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
            # Re-crop the union box from the source line. (This used to keep only the larger of the two
            # parts' crops, so the image no longer matched its own bbox.)
            full = binary[ny0:ny1, nx0:nx1]
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
