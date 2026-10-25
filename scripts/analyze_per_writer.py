"""ROADMAP Task 5.2: per-writer analysis. For the baseline and glyph_vae backends (K=1 and K=5 protocol rows), correlate each
writer's CER and HWD (and writer-ID distance, for reference) with that writer's slant, stroke width and cursiveness
(mean connected components per word), all measured on the writer's REAL test lines. VATr rows are added by re-running this
after scripts/merge_vatr_rows.py.

Outputs: results/figures/per_writer_table.csv (one row per writer x backend x K), per_writer_correlations.csv (Spearman),
writer_descriptors.json (the measured descriptors), per_writer_<backend>.png (scatter plots).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.analysis.writer_features import writer_descriptors  # noqa: E402
from src.data.iam import load_iam  # noqa: E402

FIG = ROOT / "results" / "figures"
FEATURES = [("slant_deg", "slant (deg, + = right-leaning)"), ("stroke_px", "stroke width (native px)"),
            ("cursiveness", "components per word (lower = more joined)")]
OUTCOMES = [("cer", "CER"), ("hwd", "HWD"), ("wid_dist", "writer-ID distance")]
BACKENDS = ["baseline", "glyph_vae", "vatr"]


def protocol_runs(backends=BACKENDS) -> list[dict]:
    """(backend, K, run dir) of every merged protocol row of these backends, newest run per (backend, K)."""
    rows = list(csv.DictReader((ROOT / "results" / "protocol_table.csv").open(newline="", encoding="utf-8")))
    best: dict = {}
    for r in rows:
        if r["backend"] in backends and (ROOT / "results" / r["run_id"] / "metrics.json").exists():
            best[(r["backend"], int(r["K"]))] = r["run_id"]
    return [{"backend": b, "K": k, "run_id": rid} for (b, k), rid in sorted(best.items())]


def descriptors(writers: list[str], cache: Path) -> dict:
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [w for w in writers if w not in have]
    if todo:
        ds = load_iam("test", unit="lines")
        by: dict[str, list[int]] = {}
        for i, s in enumerate(ds.samples):
            by.setdefault(s.writer_id, []).append(i)
        for w in todo:
            lines = [(ds.read_native(i), ds.samples[i].transcription) for i in by[w]]
            have[w] = writer_descriptors(lines)
            print(f"descriptors {w}: {have[w]}", flush=True)
        cache.write_text(json.dumps(have, indent=2) + "\n", encoding="utf-8")
    return {w: have[w] for w in writers}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    runs = protocol_runs()
    recs = {(r["backend"], r["K"]): json.loads((ROOT / "results" / r["run_id"] / "metrics.json").read_text("utf-8"))
            for r in runs}
    writers = sorted({w["writer"] for rec in recs.values() for w in rec["writers"]})
    desc = descriptors(writers, FIG / "writer_descriptors.json")

    table, corr = [], []
    for (b, K), rec in recs.items():
        for w in rec["writers"]:
            d = desc[w["writer"]]
            table.append({"backend": b, "K": K, "writer": w["writer"], "n_lines_real": d["n_lines"],
                          **{f: d[f] for f, _ in FEATURES}, **{o: w["all"][o] for o, _ in OUTCOMES}})
    with (FIG / "per_writer_table.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(table[0]))
        wr.writeheader()
        wr.writerows(table)

    for (b, K) in recs:
        sub = [r for r in table if r["backend"] == b and r["K"] == K]
        for fk, _ in FEATURES:
            for ok, _ in OUTCOMES:
                x, y = np.array([r[fk] for r in sub], float), np.array([r[ok] for r in sub], float)
                rho, p = spearmanr(x, y)
                corr.append({"backend": b, "K": K, "feature": fk, "outcome": ok, "n_writers": len(sub),
                             "spearman_rho": round(float(rho), 4), "p_value": round(float(p), 4)})
    with (FIG / "per_writer_correlations.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(corr[0]))
        wr.writeheader()
        wr.writerows(corr)

    for b in sorted({k[0] for k in recs}):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
        for i, (ok, olab) in enumerate(OUTCOMES[:2]):
            for j, (fk, flab) in enumerate(FEATURES):
                ax = axes[i, j]
                for K, mk in ((1, "o"), (5, "s")):
                    sub = [r for r in table if r["backend"] == b and r["K"] == K]
                    if not sub:
                        continue
                    ax.scatter([r[fk] for r in sub], [r[ok] for r in sub], marker=mk, alpha=0.75, label=f"K={K}")
                    c = next(c for c in corr if (c["backend"], c["K"], c["feature"], c["outcome"]) == (b, K, fk, ok))
                    ax.annotate(f"K={K}: rho {c['spearman_rho']:+.2f} (p {c['p_value']:.2f})", xy=(0.02, 0.95 - 0.07 * (K == 5)),
                                xycoords="axes fraction", fontsize=8, va="top")
                ax.set_xlabel(flab)
                ax.set_ylabel(olab)
                ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"{b}: per-writer CER / HWD vs the writer's real-line descriptors (20-writer subset, CPU-reduced)")
        fig.savefig(FIG / f"per_writer_{b}.png", dpi=130)
        plt.close(fig)
    print("wrote", FIG / "per_writer_table.csv", FIG / "per_writer_correlations.csv")
    for c in corr:
        if c["outcome"] in ("cer", "hwd"):
            print(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
