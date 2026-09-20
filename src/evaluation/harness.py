"""Evaluation harness: run the generator on writer-disjoint data and persist everything.

Per sample (one writer, one reference line, one target line):
  1. Condition on the writer's REFERENCE line: analyze_sample -> alphabet (baseline or neural).
  2. Render the TARGET line's text (letters only, lower-case) with that alphabet.
  3. LEGIBILITY = CER of the CNN recognizer reading the rendered image back:
     segment the rendered image (no oracle glyph boxes), classify each glyph with
     `CNNRecognizer`, compare to the target letters (spaces dropped) via edit distance.
     This is end-to-end legibility: it includes segmentation of the rendered line, so
     overlapping or touching generated letters are penalised too.
  4. STYLE FIDELITY = cosine distance between a writer-ID embedding of the generated
     image and of the reference image (and, as a second reading, of a held-out real
     line by the same writer that the generator never saw). Lower is better.
  5. Leave-one-out SSIM on the reference sample (secondary, leak-free).

Legibility and fidelity are computed together or not at all: a sample that yields
only one of them is recorded as failed, and an aggregate is never written from a
single metric. Trivial legibility (block letters) and trivial fidelity (copy the
reference) are both easy, so neither may be quoted alone.

Every run is persisted to results/<utc-timestamp>_<git-sha>/metrics.json and one
row is appended to results/index.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.evaluation.loo import leave_one_out_ssim
from src.evaluation.text_metrics import cer, levenshtein
from src.evaluation.writer_id import WriterEmbedder
from src.pipeline import analyze_sample, generate_alphabet_baseline, generate_alphabet_neural
from src.preprocessing.pipeline import Preprocessor
from src.recognition.base import Recognizer
from src.renderer.text_renderer import TextRenderer
from src.segmentation.segmenter import Segmenter
from src.utils.config import PROJECT_ROOT
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
INDEX_COLUMNS = [
    "run_id", "timestamp_utc", "git_sha", "git_dirty", "split", "unit", "backend", "is_smoke_test",
    "n_samples", "n_failed", "n_writers",
    "cer_micro", "cer_mean", "style_dist_reference_mean", "style_dist_heldout_mean", "loo_ssim_mean",
    "writer_embedder", "embedder_is_stub", "cnn_checkpoint_sha256", "vae_checkpoint_sha256", "metrics_path",
]


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info() -> dict:
    """{sha, dirty, branch}. `sha` is 'nogit' outside a repo or before the first commit."""
    sha = _git("rev-parse", "--short=8", "HEAD") or "nogit"
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "dirty": bool(status), "branch": _git("rev-parse", "--abbrev-ref", "HEAD")}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_record(path: Path | str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    return {"path": str(path), "exists": p.exists(), "sha256": file_sha256(p)}


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def letters_only(text: str) -> str:
    return "".join(c for c in text.lower() if "a" <= c <= "z")


def render_text_clean(text: str) -> str:
    """Lower-case letters and single spaces only (the alphabet has nothing else)."""
    return re.sub(r" +", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()


def truncate_text(text: str, max_chars: int) -> str:
    text = render_text_clean(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return (cut.rsplit(" ", 1)[0] if " " in cut and text[max_chars] != " " else cut).strip()


def to_binary_ink(gray: np.ndarray) -> np.ndarray:
    """Rendered/gray image -> strict 0/255 ink-on-black (what Segmenter expects)."""
    return ((np.asarray(gray) > 127).astype(np.uint8)) * 255


@dataclass
class EvalContext:
    config: dict
    backend: str                      # "baseline" | "neural"
    recognizer: Recognizer            # CNN letter classifier
    embedder: WriterEmbedder
    neural_checkpoint: str | None = None
    neural_engine: object | None = None   # StyleEncoderInference, loaded once


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray


def evaluate_pair(ctx: EvalContext, reference_gray: np.ndarray, reference_text: str, target_text: str,
                  heldout_gray: np.ndarray | None, heldout_text: str | None = None) -> dict:
    """Score one (reference sample -> generated target text). Raises if EITHER headline metric can't be computed."""
    cfg = ctx.config
    label = letters_only(reference_text)
    target = render_text_clean(target_text)
    target_letters = target.replace(" ", "")
    if not label:
        raise ValueError("reference transcription has no letters")
    if not target_letters:
        raise ValueError("target transcription has no letters")

    reference_bgr = _gray_to_bgr(reference_gray)
    analysis = analyze_sample(reference_bgr, label, cfg)

    if ctx.backend == "neural":
        alphabet = generate_alphabet_neural(analysis, checkpoint_path=ctx.neural_checkpoint, config=cfg)
    else:
        alphabet = generate_alphabet_baseline(analysis, cfg)

    rendered = to_binary_ink(TextRenderer(alphabet, cfg).render_line(target))

    # (a) legibility: CNN reads the rendered image back.
    glyphs = Segmenter(cfg).segment(rendered).glyphs
    decoded = "".join(ctx.recognizer.recognize([g.image for g in glyphs])) if glyphs else ""
    edits = levenshtein(target_letters, decoded)
    sample_cer = cer(target_letters, decoded)

    # (b) style fidelity: writer-ID embedding distance, generated vs reference (and vs a held-out real line).
    pre = Preprocessor(cfg)
    reference_binary = pre.process(image_bgr=reference_bgr, n_chars=len(label)).cropped
    dist_ref = ctx.embedder.distance(rendered, reference_binary)
    dist_held = None
    if heldout_gray is not None:
        held_chars = len(letters_only(heldout_text if heldout_text is not None else target_text))
        dist_held = ctx.embedder.distance(
            rendered, pre.process(image_bgr=_gray_to_bgr(heldout_gray), n_chars=held_chars).cropped)

    loo = leave_one_out_ssim(analysis, cfg, backend=ctx.backend, engine=ctx.neural_engine)

    return {
        "target_text": target,
        "reference_text": reference_text,
        "decoded_text": decoded,
        "n_target_letters": len(target_letters),
        "n_glyphs_detected": len(glyphs),
        "edit_distance": edits,
        "cer": sample_cer,
        "style_distance_reference": dist_ref,
        "style_distance_heldout": dist_held,
        "loo_ssim_mean": float(np.mean(list(loo.values()))) if loo else None,
        "loo_ssim_per_char": {k: float(v) for k, v in loo.items()},
        "n_reference_glyphs": len(analysis.segmentation.glyphs),
    }


def _safe_evaluate(ctx, meta: dict, *args) -> dict:
    try:
        return {**meta, **evaluate_pair(ctx, *args), "error": None}
    except Exception as e:  # noqa: BLE001 -- a bad sample must not abort the run; it is recorded and counted
        logger.warning("Sample %s failed: %s", meta.get("target_id"), e)
        return {**meta, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=4)}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def select_writers(dataset, max_writers: int | None, seed: int) -> list[str]:
    """Deterministic subset of a split's writers (hash-ranked, not just the lowest IDs)."""
    writers = sorted(dataset.writer_ids)
    if max_writers is None or max_writers >= len(writers):
        return writers
    ranked = sorted(writers, key=lambda w: hashlib.sha256(f"{seed}:eval:{w}".encode()).hexdigest())
    return sorted(ranked[:max_writers])


def run_iam_split(ctx: EvalContext, dataset, max_writers: int | None = None) -> tuple[list[dict], list[dict]]:
    """Evaluate a writer-disjoint IAM split. Returns (per_sample_records, skipped_writers)."""
    ev = ctx.config["evaluation"]
    seed = ctx.config["data"]["iam"]["split_seed"]
    by_writer: dict[str, list[int]] = {}
    for i, s in enumerate(dataset.samples):
        if len(letters_only(s.transcription)) >= ev["min_reference_letters"]:
            by_writer.setdefault(s.writer_id, []).append(i)

    records, skipped = [], []
    for writer in select_writers(dataset, max_writers, seed):
        idx = by_writer.get(writer, [])
        if len(idx) < 2:
            skipped.append({"writer_id": writer, "reason": f"only {len(idx)} usable line(s); need a reference + a target"})
            continue
        ref_i, target_idxs = idx[0], idx[1:1 + ev["targets_per_writer"]]
        ref = dataset.samples[ref_i]
        reference_gray = dataset.read_native(ref_i)
        for t_i in target_idxs:
            tgt = dataset.samples[t_i]
            meta = {"writer_id": writer, "reference_id": ref.sample_id, "target_id": tgt.sample_id}
            records.append(_safe_evaluate(
                ctx, meta, reference_gray, ref.transcription,
                truncate_text(tgt.transcription, ev["max_target_chars"]), dataset.read_native(t_i), tgt.transcription,
            ))
    return records, skipped


def run_smoke(ctx: EvalContext, samples_dir: Path) -> list[dict]:
    """Smoke test on the committed synthetic samples. NOT an evaluation: no writers, no split, no held-out image."""
    texts = ctx.config["evaluation"]["smoke_texts"]
    records = []
    for i, path in enumerate(sorted(samples_dir.glob("*_sample.png"))):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        word = path.stem.replace("_sample", "")
        meta = {"writer_id": f"synthetic:{word}", "reference_id": path.name, "target_id": f"smoke-{i}"}
        records.append(_safe_evaluate(ctx, meta, gray, word, texts[i % len(texts)], None))
    return records


# ---------------------------------------------------------------------------
# Aggregation and persistence
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return float(np.std(xs)) if xs else None


def aggregate(records: list[dict]) -> dict:
    """Aggregate over successful samples. Refuses to produce a one-metric summary."""
    ok = [r for r in records if r.get("error") is None]
    for r in ok:
        if r.get("cer") is None or r.get("style_distance_reference") is None:
            raise RuntimeError(f"sample {r.get('target_id')} is missing a headline metric; refusing to aggregate")
    if not ok:
        raise RuntimeError("No sample produced both legibility and style-fidelity metrics; nothing to aggregate. "
                           f"Errors: {[r.get('error') for r in records][:3]}")
    total_edits = sum(r["edit_distance"] for r in ok)
    total_chars = sum(r["n_target_letters"] for r in ok)
    return {
        "n_samples": len(ok),
        "n_failed": len(records) - len(ok),
        "n_writers": len({r["writer_id"] for r in ok}),
        # legibility
        "cer_micro": total_edits / total_chars,
        "cer_mean": _mean(r["cer"] for r in ok),
        "cer_std": _std(r["cer"] for r in ok),
        # style fidelity
        "style_distance_reference_mean": _mean(r["style_distance_reference"] for r in ok),
        "style_distance_reference_std": _std(r["style_distance_reference"] for r in ok),
        "style_distance_heldout_mean": _mean(r["style_distance_heldout"] for r in ok),
        # secondary
        "loo_ssim_mean": _mean(r["loo_ssim_mean"] for r in ok),
        "n_loo": sum(r["loo_ssim_mean"] is not None for r in ok),
        "glyph_count_match_rate": float(np.mean([r["n_glyphs_detected"] == r["n_target_letters"] for r in ok])),
    }


def _assert_finite(obj, path="metrics"):
    """metrics.json must be valid JSON: no NaN/inf anywhere."""
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite(v, f"{path}[{i}]")


def _unique_dir(results_dir: Path, run_id: str) -> tuple[Path, str]:
    candidate, n = results_dir / run_id, 1
    while candidate.exists():
        n += 1
        candidate = results_dir / f"{run_id}-{n}"
    return candidate, candidate.name


def write_run(results_dir: Path, record: dict) -> Path:
    """Write results/<run_id>/metrics.json and append a row to results/index.csv. Returns the metrics path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    git = record["git"]
    base_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{git['sha']}{'-dirty' if git['dirty'] else ''}"
    run_dir, run_id = _unique_dir(results_dir, base_id)
    record = {**record, "run_id": run_id, "timestamp_utc": now.isoformat(timespec="seconds")}
    _assert_finite(record)

    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    agg, ckpts = record["aggregate"], record["checkpoints"]
    row = {
        "run_id": run_id, "timestamp_utc": record["timestamp_utc"], "git_sha": git["sha"], "git_dirty": git["dirty"],
        "split": record["split"], "unit": record["unit"], "backend": record["backend"],
        "is_smoke_test": record["is_smoke_test"],
        "n_samples": agg["n_samples"], "n_failed": agg["n_failed"], "n_writers": agg["n_writers"],
        "cer_micro": agg["cer_micro"], "cer_mean": agg["cer_mean"],
        "style_dist_reference_mean": agg["style_distance_reference_mean"],
        "style_dist_heldout_mean": agg["style_distance_heldout_mean"], "loo_ssim_mean": agg["loo_ssim_mean"],
        "writer_embedder": record["writer_embedder"]["name"], "embedder_is_stub": record["writer_embedder"]["is_stub"],
        "cnn_checkpoint_sha256": (ckpts["cnn"] or {}).get("sha256"),
        "vae_checkpoint_sha256": (ckpts["vae"] or {}).get("sha256"),
        "metrics_path": _display_path(metrics_path),
    }
    append_index_row(results_dir / "index.csv", row)
    return metrics_path


def _display_path(path: Path) -> str:
    """Repo-relative (forward slashes) when inside the repo, absolute otherwise."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def append_index_row(index_path: Path, row: dict) -> None:
    exists = index_path.exists() and index_path.stat().st_size > 0
    if exists:
        with index_path.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != INDEX_COLUMNS:
            raise ValueError(f"{index_path} has columns {header}, expected {INDEX_COLUMNS}. Move it aside "
                             "(the per-run metrics.json files are unaffected) instead of mixing schemas.")
    with index_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: ("" if row[k] is None else row[k]) for k in INDEX_COLUMNS})
