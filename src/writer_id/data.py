"""Writer-ID data: canonical crops, an on-disk crop cache, window sampling and augmentation.

Canonical model input = a uint8 image, INK HIGH (255 = ink, 0 = paper), height `input_height`,
width scaled to preserve aspect ratio. Both sides of the style-fidelity comparison reach it the same way:

  IAM gray scan  --Otsu at native resolution-->  binary ink=255  --area resize to height H-->  crop
  generated / preprocessed binary line (ink=255) ------------------ area resize to height H-->  crop

so a generated render and a real reference are compared in the same domain.

Augmentation is affine (tiny rotation, isotropic scale, translation) plus elastic distortion. There is
deliberately NO shear/slant change and NO stroke-width change (dilate/erode/blur): slant and stroke width
are the style signal the model is supposed to encode.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.leakage import assert_no_test_writers
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# -- canonicalisation -----------------------------------------------------------------------------

def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA if height < h else cv2.INTER_CUBIC)


def canonical_from_iam_gray(gray: np.ndarray, height: int, binarize: bool = True) -> np.ndarray:
    """IAM scan crop (dark ink on light paper) -> canonical ink-high crop."""
    if binarize:
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ink = 255 - gray
    return resize_to_height(ink, height)


def canonical_from_binary(binary_ink: np.ndarray, height: int) -> np.ndarray:
    """Pipeline binary image (ink=255 on 0, any height) -> canonical crop."""
    return resize_to_height(((np.asarray(binary_ink) > 127).astype(np.uint8)) * 255, height)


# -- on-disk cache -----------------------------------------------------------------------------------

class LineStore:
    """Decoded, canonicalised IAM lines cached as .npy files and read back memory-mapped (keeps RAM flat)."""

    def __init__(self, cache_dir: Path, height: int, binarize: bool):
        self.dir = Path(cache_dir) / f"h{height}_{'bin' if binarize else 'gray'}"
        self.height, self.binarize = height, binarize
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sample_id: str) -> Path:
        return self.dir / f"{sample_id}.npy"

    def ensure(self, dataset) -> None:
        """Write any missing crops for `dataset` (an IAMDataset)."""
        missing = [i for i, s in enumerate(dataset.samples) if not self._path(s.sample_id).exists()]
        for n, i in enumerate(missing):
            crop = canonical_from_iam_gray(dataset.read_native(i), self.height, self.binarize)
            np.save(self._path(dataset.samples[i].sample_id), crop)
            if (n + 1) % 1000 == 0:
                logger.info("cached %d/%d %s lines", n + 1, len(missing), dataset.name)
        if missing:
            logger.info("cached %d new lines under %s", len(missing), self.dir)

    def load(self, sample_id: str) -> np.ndarray:
        return np.load(self._path(sample_id), mmap_mode="r")


# -- windows and augmentation -----------------------------------------------------------------------------

def random_window(line: np.ndarray, width: int, rng: np.random.Generator, min_ink: float = 0.004, tries: int = 6) -> np.ndarray:
    """A random `width`-px window of a line (aspect preserved). Retries a few times to avoid blank margins;
    lines shorter than `width` are right-padded with paper."""
    h, w = line.shape
    if w <= width:
        out = np.zeros((h, width), dtype=np.uint8)
        out[:, :w] = line
        return out
    best = None
    for _ in range(tries):
        x0 = int(rng.integers(0, w - width + 1))
        win = np.asarray(line[:, x0:x0 + width])
        best = win if best is None or win.mean() > best.mean() else best
        if (win > 127).mean() >= min_ink:
            return win
    return best


def _elastic_field(shape, alpha: float, sigma: float, rng: np.random.Generator):
    h, w = shape
    k = int(sigma * 4) | 1
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (k, k), sigma) * alpha * 4
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (k, k), sigma) * alpha * 4
    return dx, dy


def augment_window(win: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Small affine jitter + elastic distortion. `cfg` is the `writer_id.augment` block.

    Rotation is limited to ~1 deg (it perturbs the measured slant by that much, zero-mean); there is no
    shear term and nothing that thickens or thins strokes.

    The affine and the elastic field are composed into ONE sampling map and applied with a single `remap`.
    Applying them as two successive interpolations (warpAffine, then remap) blurs the ink twice, which fattens
    the anti-aliased fringe of every stroke by ~1 px: it leaves thresholded stroke width and ink mass unchanged
    but moves the measured stroke width by ~17%, i.e. it leaks a stroke-width change into the style signal.
    """
    h, w = win.shape
    angle = rng.uniform(-cfg["rotation_deg"], cfg["rotation_deg"])
    scale = 1.0 + rng.uniform(-cfg["scale"], cfg["scale"])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += rng.uniform(-cfg["translate_px"], cfg["translate_px"])
    m[1, 2] += rng.uniform(-cfg["translate_px"], cfg["translate_px"])
    inv = cv2.invertAffineTransform(m)                       # output pixel -> source pixel
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    if rng.random() < cfg["elastic_prob"]:
        dx, dy = _elastic_field(win.shape, cfg["elastic_alpha"], cfg["elastic_sigma"], rng)
        gx, gy = gx + dx, gy + dy
    map_x = (inv[0, 0] * gx + inv[0, 1] * gy + inv[0, 2]).astype(np.float32)
    map_y = (inv[1, 0] * gx + inv[1, 1] * gy + inv[1, 2]).astype(np.float32)
    return cv2.remap(win, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class WriterBatchSampler:
    """P writers x K crops per batch, so every anchor has K-1 same-writer positives for the contrastive term."""

    def __init__(self, store: LineStore, samples_by_writer: dict[str, list], writer_to_idx: dict[str, int],
                 writers_per_batch: int, crops_per_writer: int, aug_cfg: dict, seed: int,
                 forbidden_writers=frozenset()):
        self.store, self.by_writer, self.idx = store, samples_by_writer, writer_to_idx
        self.forbidden = frozenset(forbidden_writers)          # TEST-split writers: must never reach a batch
        assert_no_test_writers(samples_by_writer, self.forbidden, "WriterBatchSampler's writer pool")
        self.writers = sorted(w for w in samples_by_writer if samples_by_writer[w] and w in writer_to_idx)
        self.P, self.K, self.aug = writers_per_batch, crops_per_writer, aug_cfg
        self.rng = np.random.default_rng(seed)
        if len(self.writers) < 2:
            raise ValueError("need at least 2 training writers")

    def sample(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        chosen = self.rng.choice(self.writers, size=min(self.P, len(self.writers)), replace=False)
        assert_no_test_writers(chosen, self.forbidden)
        wins, labels = [], []
        for w in chosen:
            samples = self.by_writer[w]
            for _ in range(self.K):
                s = samples[int(self.rng.integers(len(samples)))]
                win = random_window(self.store.load(s.sample_id), width, self.rng)
                wins.append(augment_window(win, self.aug, self.rng))
                labels.append(self.idx[w])
        return np.stack(wins), np.asarray(labels, dtype=np.int64)
