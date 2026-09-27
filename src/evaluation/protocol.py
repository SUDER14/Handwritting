"""ROADMAP Task 1.4 evaluation protocol: K reference lines -> generate the writer's OTHER lines -> score.

For each TEST writer (a fixed, seeded subset -- CPU budget) and each K:
  * K reference lines are drawn (seeded, from lines with enough letters); the writer's remaining lines are the
    held-out real lines; up to `targets_per_writer` of them (seeded) become generation targets.
  * a backend generates one line image per target text from the K references;
  * per writer we score  legibility  = CER of the frozen CRNN reading the generated lines,
                          fidelity    = (a) cosine distance between the mean writer-ID embedding of the generated
                                         lines and that of the writer's real held-out lines,
                                         (b) HWD(generated set, real held-out set)  (official HWD, see hwd_metric.py),
                          and the same three on the OOV SUBSET = target lines containing a letter absent from the
                          K reference transcriptions;
  * the "real" upper-bound row runs REAL held-out lines through the identical scoring: half of the targets play the
    role of the "generated" set and are compared with the writer's other real held-out lines (a disjoint half).
Numbers are averaged per writer, then over writers. Text fed to the generators is lower-case letters + spaces
(`harness.render_text_clean`, truncated to `max_target_chars`); CER compares the recognizer output normalised the same way.
Test-writer leakage: nothing here trains; the trained instruments were fitted on train writers only (asserted there).
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.evaluation.harness import letters_only, render_text_clean, truncate_text
from src.evaluation.hwd_metric import hwd, set_mean
from src.evaluation.text_metrics import levenshtein
from src.utils.logging_setup import get_logger
from src.writer_id.data import canonical_from_iam_gray

logger = get_logger(__name__)


@dataclass
class Ref:
    gray: np.ndarray        # uint8, dark ink on light paper (IAM native polarity)
    text: str


class Backend(Protocol):
    name: str

    def generate(self, refs: list[Ref], text: str) -> np.ndarray | None:
        """K reference lines + target text -> uint8 gray line image, dark ink on light paper (None = failed)."""


_OOM_MARKERS = ("insufficient memory", "not enough memory", "out of memory", "failed to allocate", "bad_alloc")


def is_out_of_memory(e: BaseException) -> bool:
    """Host-RAM exhaustion, whatever library raised it: Python MemoryError, torch's DefaultCPUAllocator RuntimeError,
    OpenCV's cv2.error (-4: Insufficient memory). These are infrastructure failures, not generation failures."""
    return isinstance(e, MemoryError) or any(m in str(e).lower() for m in _OOM_MARKERS)


def choose_writers(dataset, n_writers: int | None, seed: int, min_lines: int) -> list[str]:
    """A deterministic subset of the split's writers that have enough lines for K refs + targets."""
    by: dict[str, int] = {}
    for s in dataset.samples:
        by[s.writer_id] = by.get(s.writer_id, 0) + 1
    ok = sorted(w for w, n in by.items() if n >= min_lines)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(ok)))
    picked = [ok[i] for i in order]
    return sorted(picked[:n_writers] if n_writers else picked)


def split_lines(samples, K: int, targets_per_writer: int, min_letters: int, seed: int, writer: str):
    """-> (reference samples, target samples). Seeded per (writer, K); references/targets are disjoint lines."""
    rng = np.random.default_rng([seed, int(writer) if writer.isdigit() else sum(writer.encode()), K])
    eligible = [s for s in samples if len(letters_only(s.transcription)) >= min_letters]
    if len(eligible) < K + 1:
        return None
    refs = [eligible[i] for i in rng.permutation(len(eligible))[:K]]
    ref_ids = {r.sample_id for r in refs}
    rest = [s for s in samples if s.sample_id not in ref_ids and len(letters_only(s.transcription)) >= 3]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    return refs, rest[:targets_per_writer]


def is_oov(target_text: str, ref_texts: list[str]) -> bool:
    seen = set("".join(letters_only(t) for t in ref_texts))
    return any(c not in seen for c in letters_only(target_text))


class Scorer:
    """The frozen instruments: recognizer (legibility), writer-ID embedder + HWD features (fidelity)."""

    def __init__(self, recognizer, embedder, hwd_feats, height: int = 64, hwd_cache=None):
        self.recognizer, self.embedder, self.hwd_feats, self.height = recognizer, embedder, hwd_feats, height
        self.hwd_cache = hwd_cache if hwd_cache is not None else {}

    def crop(self, gray: np.ndarray) -> np.ndarray:
        return canonical_from_iam_gray(np.asarray(gray, np.uint8), self.height)

    def read(self, grays: list[np.ndarray]) -> list[str]:
        return self.recognizer.transcribe([self.crop(g) for g in grays])

    def embed(self, gray: np.ndarray) -> np.ndarray:
        return self.embedder.embed(self.crop(gray)).astype(np.float64)

    def scorable(self, gray: np.ndarray) -> bool:
        """Ink survives canonicalisation (Otsu + resize to `height`): what embed() requires. A generation with a few
        faint/thin marks can pass the raw-pixel check yet vanish here; it is then a failed generation, not a lost writer."""
        return bool(np.any(self.crop(gray) > 127))

    def hwd_entry(self, gray: np.ndarray, key: str | None = None):
        if key is not None and key in self.hwd_cache:
            return self.hwd_cache[key]
        e = self.hwd_feats.line(gray)
        if key is not None:
            self.hwd_cache[key] = e
        return e


def _cosine_dist_of_means(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    def unit_mean(x):
        m = np.mean([v / np.linalg.norm(v) for v in x], axis=0)
        return m
    ma, mb = unit_mean(a), unit_mean(b)
    return float(1.0 - np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb)))


def score_set(scorer: Scorer, gen_grays, gen_texts, real_grays, real_keys, gen_keys=None) -> dict:
    """Metrics of one set of lines (`gen_*`) against the writer's real held-out lines (`real_*`)."""
    hyps = [render_text_clean(h) for h in scorer.read(gen_grays)]
    refs = [render_text_clean(t) for t in gen_texts]
    edits = sum(levenshtein(r, h) for r, h in zip(refs, hyps))
    n_ref = sum(len(r) for r in refs)
    g_emb = [scorer.embed(g) for g in gen_grays]
    r_emb = [scorer.embed(g) for g in real_grays]
    g_h = [scorer.hwd_entry(g, k) for g, k in zip(gen_grays, gen_keys or [None] * len(gen_grays))]
    r_h = [scorer.hwd_entry(g, k) for g, k in zip(real_grays, real_keys)]
    return {"cer": edits / max(n_ref, 1), "wid_dist": _cosine_dist_of_means(g_emb, r_emb),
            "hwd": hwd(set_mean(g_h), set_mean(r_h)), "n_lines": len(gen_grays), "hyps": hyps}


def run_writer(scorer: Scorer, backend: Backend | None, dataset, samples, K: int, cfg: dict, seed: int, writer: str):
    """One writer, one K, one backend (None = the "real" row). Returns a dict or None if the writer can't be scored."""
    sp = split_lines(samples, K, cfg["targets_per_writer"], cfg["min_reference_letters"], seed, writer)
    if sp is None:
        return None
    refs, targets = sp
    if len(targets) < 2:
        return None
    index = {s.sample_id: i for i, s in enumerate(dataset.samples)}
    real = [dataset.read_native(index[t.sample_id]) for t in targets]
    keys = [t.sample_id for t in targets]
    ref_texts = [r.text if isinstance(r, Ref) else r.transcription for r in refs]
    oov_flags = [is_oov(t.transcription, ref_texts) for t in targets]
    texts = [truncate_text(t.transcription, cfg["max_target_chars"]) for t in targets]
    out = {"writer": writer, "K": K, "n_targets": len(targets), "ref_ids": [r.sample_id for r in refs],
           "target_ids": keys, "oov_flags": oov_flags}
    if backend is None:                                          # REAL row: half the real lines vs the other half
        half = len(targets) // 2
        gen_idx, ref_idx = list(range(half)), list(range(half, len(targets)))
        full = [render_text_clean(t.transcription) for t in targets]
        m = score_set(scorer, [real[i] for i in gen_idx], [full[i] for i in gen_idx],
                      [real[i] for i in ref_idx], [keys[i] for i in ref_idx], [keys[i] for i in gen_idx])
        out["all"] = m
        oov_idx = [i for i in gen_idx if oov_flags[i]]
        out["oov"] = (score_set(scorer, [real[i] for i in oov_idx], [full[i] for i in oov_idx],
                                [real[i] for i in ref_idx], [keys[i] for i in ref_idx], [keys[i] for i in oov_idx])
                      if oov_idx else None)
        return out
    ref_objs = [Ref(dataset.read_native(index[r.sample_id]), r.transcription) for r in refs]
    gens, ok, errors = [], [], []
    for t in texts:
        g = None
        for attempt in range(2):
            try:
                g = backend.generate(ref_objs, t) if t.strip() else None
                break
            except Exception as e:                               # noqa: BLE001 -- a failed generation is counted
                if is_out_of_memory(e):                          # host out of RAM: infrastructure, retry once
                    errors.append(f"OutOfMemory[{type(e).__name__}](attempt {attempt + 1}): {e}")
                    logger.warning("writer %s K=%d: out of memory generating %r (attempt %d): %r",
                                   writer, K, t[:30], attempt + 1, e)
                    gc.collect()
                    continue
                errors.append(f"{type(e).__name__}: {e}")
                logger.warning("writer %s K=%d: generation failed for %r: %r", writer, K, t[:30], e)
                break
        gens.append(g)
        ok.append(g is not None and bool(np.any(np.asarray(g) < 128))
                  and getattr(scorer, "scorable", lambda _g: True)(g))
    out["n_failed"] = int(len(ok) - sum(ok))
    out["gen_errors"] = errors
    keep = [i for i, o in enumerate(ok) if o]
    if len(keep) < 1:
        logger.warning("writer %s K=%d: every generation failed (%s) -> writer skipped", writer, K, errors[:3])
        return None
    out["all"] = score_set(scorer, [gens[i] for i in keep], [texts[i] for i in keep], real, keys)
    oov = [i for i in keep if oov_flags[i]]
    out["oov"] = (score_set(scorer, [gens[i] for i in oov], [texts[i] for i in oov], real, keys) if oov else None)
    out["gen_examples"] = [(texts[i], out["all"]["hyps"][j]) for j, i in enumerate(keep[:3])]
    return out


def aggregate(rows: list[dict]) -> dict:
    """Mean over writers of the per-writer metrics (writers with an OOV subset counted for the OOV means)."""
    def mean(key, sub):
        v = [r[sub][key] for r in rows if r.get(sub)]
        return (float(np.mean(v)), len(v)) if v else (None, 0)
    out = {"n_writers": len(rows)}
    for sub in ("all", "oov"):
        for key in ("cer", "wid_dist", "hwd"):
            m, n = mean(key, sub)
            out[f"{sub}_{key}"] = m
        out[f"{sub}_n_writers"] = mean("cer", sub)[1]
        out[f"{sub}_n_lines"] = int(sum(r[sub]["n_lines"] for r in rows if r.get(sub)))
    out["n_failed_generations"] = int(sum(r.get("n_failed", 0) for r in rows))
    return out
