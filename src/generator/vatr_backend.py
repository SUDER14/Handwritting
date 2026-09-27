"""VATr (Pippi et al., CVPR 2023) as a protocol backend (ROADMAP Task 3.3), same interface as baseline / glyph_vae:

    VATrBackend(cfg).generate(refs: list[Ref], text: str) -> uint8 gray line image, dark ink on light paper

VATr's style input is 15 single-WORD images (height 32, width <= 192). IAM word boxes are NOT available here (the
Teklia-joined lines.txt has zeroed boxes and there is no words.txt), so the references are adapted (logged in
ROADMAP_STATE.md as a Stage 3 adaptation):

  1. each reference LINE is split into words at its N-1 widest ink gaps, N = number of words in its transcription
     (a token with no letter/digit, e.g. a lone '.', belongs to the previous word: it is written attached to it);
  2. each word crop is trimmed to its own ink box (IAM word images are tight boxes, which is what VATr was trained on);
  3. the crops of all K references are cycled, in order, up to VATr's 15 style slots (for K=1 a line gives ~5-9 words,
     so words repeat; VATr's own FolderDataset samples 15 distinct files and cannot take fewer).

Style preprocessing then replicates third_party/VATr/data/dataset.py::FolderDataset.sample_style exactly (resize to
height 32 keeping aspect, right-pad / crop to width 192 with paper, ToTensor, Normalize(0.5, 0.5)). The line is
generated as VATr's generator.py does: one image per word, joined with 16 px paper gaps. Nothing in third_party/VATr is
edited; runtime shims (wandb stub, unused FID InceptionV3 stub, map_location) are the same as scripts/vatr_reproduce.py.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np

from src.utils.config import resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

NUM_STYLE = 15          # VATr num_examples
STYLE_H = 32            # VATr img_height
STYLE_MAX_W = 192       # FolderDataset max_width
WORD_GAP = 16           # generator.py: np.ones((32, 16)) between words


# ------------------------------------------------------------------ reference lines -> word crops (no GPU needed)

def transcription_words(text: str) -> list[str]:
    """Whitespace tokens, with letter/digit-free tokens (punctuation) merged into the previous word."""
    words: list[str] = []
    for tok in text.split():
        if words and not re.search(r"[A-Za-z0-9]", tok):
            words[-1] += tok
        else:
            words.append(tok)
    return words


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Dark-ink-on-light uint8 -> bool ink mask (Otsu)."""
    g = np.asarray(gray, dtype=np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0


def _trim(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def split_line_into_words(gray: np.ndarray, n_words: int) -> list[np.ndarray]:
    """Split a line image at its n_words-1 widest ink-free column gaps; each piece trimmed to its ink box.

    Returns fewer than n_words crops when the line has fewer gaps (touching words) -- never invents a split inside ink.
    """
    g = np.asarray(gray, dtype=np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    mask = ink_mask(g)
    cols = mask.any(axis=0)
    xs = np.nonzero(cols)[0]
    if len(xs) == 0:
        return []
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    gaps = []                                           # (width, start, end) of empty-column runs inside [x0, x1)
    x = x0
    while x < x1:
        if not cols[x]:
            s = x
            while x < x1 and not cols[x]:
                x += 1
            gaps.append((x - s, s, x))
        else:
            x += 1
    k = max(0, min(n_words - 1, len(gaps)))
    cuts = sorted((s + e) // 2 for _, s, e in sorted(gaps, key=lambda t: (-t[0], t[1]))[:k])
    bounds = [x0, *cuts, x1]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = _trim(g[:, a:b], mask[:, a:b])
        if c is not None:
            out.append(c)
    return out


def reference_word_crops(refs) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for r in refs:
        n = len(transcription_words(r.text))
        pieces = split_line_into_words(r.gray, max(n, 1))
        if len(pieces) != n:
            logger.info("reference line split into %d pieces for %d transcription words", len(pieces), n)
        crops += pieces
    return crops


def cycle_to(items: list, n: int) -> list:
    if not items:
        raise ValueError("no word crops in the references")
    return [items[i % len(items)] for i in range(n)]


def style_tensor_array(crops: list[np.ndarray]) -> np.ndarray:
    """FolderDataset.sample_style preprocessing, without the random draw: (15, 32, 192) float32 in [-1, 1]."""
    out = np.empty((len(crops), STYLE_H, STYLE_MAX_W), np.float32)
    for i, c in enumerate(crops):
        h, w = c.shape
        from PIL import Image
        img = np.array(Image.fromarray(c).resize((max(1, w * STYLE_H // h), STYLE_H), Image.BILINEAR))
        inv = 255 - img
        pad = np.zeros((STYLE_H, STYLE_MAX_W), np.float32)
        pad[:, : inv.shape[1]] = inv[:, :STYLE_MAX_W]
        pad = (255 - pad).astype(np.uint8)
        out[i] = (pad.astype(np.float32) / 255.0 - 0.5) / 0.5          # ToTensor + Normalize((0.5,), (0.5,))
    return out


# ------------------------------------------------------------------ the model (needs third_party/VATr + files/vatr.pth)

@contextlib.contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _import_vatr(vatr_root: Path):
    """Import VATr's generator module with the logged runtime shims; returns the module."""
    import torch
    import torch.nn as nn
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    if str(vatr_root) not in sys.path:
        sys.path.insert(0, str(vatr_root))
    with _cwd(vatr_root):
        import models.inception as inc

        class _NoInception(nn.Module):            # FID network: built in VATr.__init__, never used for generation
            BLOCK_INDEX_BY_DIM = inc.InceptionV3.BLOCK_INDEX_BY_DIM

            def __init__(self, *a, **k):
                super().__init__()

        inc.InceptionV3 = _NoInception
        import models.model as mm
        mm.InceptionV3 = _NoInception
        import generator as gen
    return gen, torch


class VATrBackend:
    name = "vatr"

    def __init__(self, config: dict, checkpoint: str | None = None, vatr_root: str | None = None,
                 device: str | None = None, seed: int = 0):
        vc = config.get("vatr", {})
        self.vatr_root = Path(vatr_root or resolve_path(vc.get("root", "third_party/VATr")))
        self.checkpoint = Path(checkpoint or (self.vatr_root / vc.get("checkpoint", "files/vatr.pth")))
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"VATr checkpoint not found: {self.checkpoint} (see notebooks/colab_train.ipynb)")
        gen, torch = _import_vatr(self.vatr_root)
        self._torch = torch
        args = gen.FakeArgs()
        args.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args.add_noise = False
        self.device = args.device
        torch.manual_seed(seed)
        with _cwd(self.vatr_root):                 # VATr opens files/unifont.pickle etc. relative to its root
            model = gen.VATr(args)
            ck = torch.load(self.checkpoint, map_location=args.device, weights_only=False)
            gen.load_checkpoint(model, ck)
        model.eval()
        self.model = model
        self.alphabet = set(args.alphabet)
        self.dropped_chars: dict[str, int] = {}
        self._cached: tuple | None = None
        logger.info("VATr loaded from %s on %s", self.checkpoint, self.device)

    def _style(self, refs):
        key = tuple((r.text, r.gray.shape, hash(np.ascontiguousarray(r.gray).tobytes())) for r in refs)
        if self._cached is None or self._cached[0] != key:
            crops = cycle_to(reference_word_crops(refs), NUM_STYLE)
            arr = style_tensor_array(crops)
            self._cached = (key, self._torch.from_numpy(arr)[None].to(self.device))
        return self._cached[1]

    def generate(self, refs, text: str):
        words = []
        for w in text.split():
            kept = "".join(ch for ch in w if ch in self.alphabet)
            for ch in w:
                if ch not in self.alphabet:
                    self.dropped_chars[ch] = self.dropped_chars.get(ch, 0) + 1
            if kept:
                words.append(kept)
        if not words:
            return None
        style = self._style(refs)
        with self._torch.no_grad():
            enc, lens, pos = self.model.netconverter.encode(words)
            fakes = self.model._generate_fakes(style, enc.to(self.device).unsqueeze(0), lens, pos)
        gap = np.ones((STYLE_H, WORD_GAP), np.float32)
        line = np.concatenate(sum([[f, gap] for f in fakes], [])[:-1], axis=1)
        return np.clip(line * 255.0, 0, 255).astype(np.uint8)

    def coverage(self) -> dict:
        return {"dropped_chars_not_in_vatr_alphabet": dict(sorted(self.dropped_chars.items()))}
