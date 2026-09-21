"""Embedding extraction and writer-retrieval metrics (top-1 and mAP, cosine distance).

Protocol (evaluated on writers the model never saw): every line is a query; the gallery is all OTHER
lines (and, by default, excluding lines from the query's own form/page, because a page shares pen, paper and
scan artefacts and retrieving within a page is easier than retrieving a writer). A gallery line is a positive
if it has the query's writer. Queries with no positive in their gallery are skipped.

  top-1 = fraction of queries whose nearest gallery line (highest cosine similarity) is a positive
  mAP   = mean over queries of average precision of the full gallery ranking
  chance top-1 = mean over queries of (#positives / gallery size): what random ranking would score
"""
from __future__ import annotations

import numpy as np
import torch

from src.writer_id.data import LineStore


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@torch.no_grad()
def embed_line(encoder, line: np.ndarray, eval_cfg: dict) -> np.ndarray:
    """One canonical line (uint8, ink high, H x W) -> unit-length embedding."""
    encoder.eval()
    line = np.asarray(line)
    h, w = line.shape
    if eval_cfg["mode"] == "full_line":
        cap = eval_cfg["max_line_width"]
        if w > cap:
            x0 = (w - cap) // 2
            line = line[:, x0:x0 + cap]
        x = torch.from_numpy(line.astype(np.float32) / 255.0)[None, None]
        return _l2(encoder(x).numpy()[0])

    width, stride = eval_cfg["window_width"], eval_cfg["window_stride"]
    if w <= width:
        pad = np.zeros((h, width), dtype=np.uint8)
        pad[:, :w] = line
        starts = [0]
        line = pad
    else:
        starts = list(range(0, w - width + 1, stride))
        if len(starts) > eval_cfg["max_windows"]:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, eval_cfg["max_windows"]).astype(int)]
    wins = np.stack([line[:, s:s + width] for s in starts])
    ink = (wins > 127).mean(axis=(1, 2))
    keep = ink >= 0.004 if (ink >= 0.004).any() else np.ones(len(wins), bool)   # skip blank margins unless all blank
    x = torch.from_numpy(wins[keep].astype(np.float32) / 255.0)[:, None]
    return _l2(_l2(encoder(x).numpy()).mean(axis=0))


def embed_samples(encoder, store: LineStore, samples: list, eval_cfg: dict) -> np.ndarray:
    return np.stack([embed_line(encoder, store.load(s.sample_id), eval_cfg) for s in samples])


def retrieval_metrics(emb: np.ndarray, writers: list[str], forms: list[str], exclude_same_form: bool = True) -> dict:
    """See the module docstring. `emb` need not be normalised."""
    e = _l2(np.asarray(emb, dtype=np.float64))
    w, f = np.asarray(writers), np.asarray(forms)
    sim = e @ e.T
    n = len(e)
    top1, aps, chance = [], [], []
    for i in range(n):
        gallery = np.ones(n, dtype=bool)
        gallery[i] = False
        if exclude_same_form:
            gallery &= f != f[i]
        pos = gallery & (w == w[i])
        if not pos.any():
            continue
        idx = np.nonzero(gallery)[0]
        order = idx[np.argsort(-sim[i, idx], kind="stable")]
        hit = (w[order] == w[i])
        top1.append(bool(hit[0]))
        ranks = np.nonzero(hit)[0] + 1
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
        chance.append(pos.sum() / gallery.sum())
    if not top1:
        raise ValueError("no query has a same-writer line in another form; cannot compute retrieval metrics")
    return {
        "top1": float(np.mean(top1)), "map": float(np.mean(aps)), "chance_top1": float(np.mean(chance)),
        "n_queries": len(top1), "n_lines": n, "n_writers": len(set(writers)),
        "exclude_same_form": bool(exclude_same_form),
    }
