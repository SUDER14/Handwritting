"""Stage 4 measurements (no GPU, no VATr).

    python scripts/stage4_measure.py words  --allow-test   # 4.1: learn the word-gap rule on TRAIN lines, check on
                                                           #      VAL, GATE 4 count check on 200 seeded TEST lines
    python scripts/stage4_measure.py deskew --allow-test   # 4.2: skew-estimator output distribution on the same 200 lines

Results -> results/stage4_word_segmentation.json, results/stage4_deskew.json. The 200 test lines are a seeded draw
(evaluation.protocol.seed) from the canonical test split; test lines are only READ (nothing is fitted on them).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.data.iam import load_iam_splits
from src.evaluation.harness import git_info, letters_only
from src.generator.vatr_backend import transcription_words
from src.segmentation.words import despeckle, learn_gap_rule, predicted_word_count
from src.utils.config import load_config, resolve_path
from src.writer_id.data import LineStore

N_TEST, N_TRAIN_FIT = 200, 3000


def test_lines(datasets, seed):
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(datasets["test"]), N_TEST, replace=False).tolist())
    return idx


def _lines(store, ds, idx):
    return [(despeckle(np.asarray(store.load(ds.samples[i].sample_id))), ds.samples[i].transcription) for i in idx]


def words(cfg, datasets, store, seed) -> dict:
    rng = np.random.default_rng(seed)
    tr = datasets["train"]
    tr_idx = sorted(rng.choice(len(tr), min(N_TRAIN_FIT, len(tr)), replace=False).tolist())
    train = [(ink, len(transcription_words(t))) for ink, t in _lines(store, tr, tr_idx)]
    fit = learn_gap_rule(train)
    rule = fit["rule"]

    def score(ds, idx):
        rows = []
        for i, (ink, text) in zip(idx, _lines(store, ds, idx)):
            n_words, n_tokens = len(transcription_words(text)), len(text.split())
            p = predicted_word_count(ink, rule)
            rows.append({"id": ds.samples[i].sample_id, "pred": p, "words": n_words, "tokens": n_tokens})
        w1 = float(np.mean([abs(r["pred"] - r["words"]) <= 1 for r in rows]))
        ex = float(np.mean([r["pred"] == r["words"] for r in rows]))
        tok1 = float(np.mean([abs(r["pred"] - r["tokens"]) <= 1 for r in rows]))
        err = np.array([r["pred"] - r["words"] for r in rows])
        return {"n_lines": len(rows), "within1": w1, "exact": ex, "within1_vs_raw_tokens": tok1,
                "mean_signed_error": float(err.mean()), "frac_over": float((err > 1).mean()),
                "frac_under": float((err < -1).mean()), "rows": rows}

    va = datasets["val"]
    val = score(va, list(range(len(va))))
    test = score(datasets["test"], test_lines(datasets, seed))
    return {"rule": rule, "fit": fit,
            "val": {k: v for k, v in val.items() if k != "rows"},
            "test200": {k: v for k, v in test.items() if k != "rows"}, "test200_rows": test["rows"],
            "gate4_word_count_within1_ge_0.80": test["within1"] >= 0.80}


def deskew(cfg, datasets, seed) -> dict:
    from src.preprocessing.pipeline import Preprocessor
    pp = Preprocessor(cfg)
    ds = datasets["test"]
    rows = []
    for i in test_lines(datasets, seed):
        s = ds.samples[i]
        g = ds.read_native(i)
        r = pp.process(image_bgr=cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), n_chars=len(letters_only(s.transcription)))
        rows.append({"id": s.sample_id, "angle": r.deskew_angle_deg, "skipped": r.deskew_skipped,
                     "n_chars": r.n_chars_for_deskew})
    a = np.array([r["angle"] for r in rows if not r["skipped"]])
    lim = float(cfg["preprocessing"]["deskew_angle_search_deg"])
    return {"n_lines": len(rows), "n_skipped_short": int(sum(r["skipped"] for r in rows)),
            "deskew_min_chars": cfg["preprocessing"]["deskew_min_chars"], "search_deg": lim,
            "angle_mean": float(a.mean()), "angle_std": float(a.std()), "angle_median": float(np.median(a)),
            "abs_angle_percentiles": {p: float(np.percentile(np.abs(a), p)) for p in (50, 75, 90, 95, 99)},
            "frac_exactly_zero": float(np.mean(a == 0.0)), "frac_abs_ge_2": float(np.mean(np.abs(a) >= 2)),
            "frac_abs_ge_5": float(np.mean(np.abs(a) >= 5)), "frac_at_search_limit": float(np.mean(np.abs(a) >= lim)),
            "histogram_1deg": {int(k): int(v) for k, v in zip(*np.unique(np.round(a).astype(int), return_counts=True))},
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["words", "deskew"])
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args()
    if not args.allow_test:
        print("Refusing to read the test split without --allow-test.", file=sys.stderr)
        return 2
    cfg = load_config()
    seed = cfg["evaluation"]["protocol"]["seed"]
    datasets = load_iam_splits(unit="lines", config=cfg)
    head = {"task": "4.1" if args.what == "words" else "4.2", "utc": datetime.now(timezone.utc).isoformat(),
            "git": git_info(), "seed": seed}
    if args.what == "words":
        wid = cfg["writer_id"]
        store = LineStore(resolve_path(wid["cache_dir"]), 64, wid["binarize"])
        for n in ("train", "val", "test"):
            store.ensure(datasets[n])
        res = {**head, **words(cfg, datasets, store, seed)}
        out = resolve_path("results/stage4_word_segmentation.json")
        summary = {k: res[k] for k in ("rule", "fit", "val", "test200", "gate4_word_count_within1_ge_0.80")}
    else:
        res = {**head, **deskew(cfg, datasets, seed)}
        out = resolve_path("results/stage4_deskew.json")
        summary = {k: v for k, v in res.items() if k != "rows"}
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
