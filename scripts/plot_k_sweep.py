"""ROADMAP Task 5.1: legibility vs fidelity as the number of reference lines K grows (CER on x, HWD on y).

Reads the protocol rows in results/protocol_table.csv for K in {1, 3, 5, 10, 15} (baseline now; VATr is drawn too as soon as
its rows are merged -- until then it is simply absent, never invented). Because writers drop out at large K (a writer
needs K+1 eligible lines and >= 2 targets), two curves are drawn per backend:
  * all writers scored at each K ("as run": the writer set shrinks with K), and
  * the COMMON writers (scored at every K), which is the like-for-like trend.
Outputs: results/figures/k_sweep_<backend>.csv, results/figures/k_sweep.png.
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

KS = [1, 3, 5, 10, 15]
FIG = ROOT / "results" / "figures"


def load_runs(backend: str) -> dict[int, dict]:
    rows = list(csv.DictReader((ROOT / "results" / "protocol_table.csv").open(newline="", encoding="utf-8")))
    out = {}
    for r in rows:
        if r["backend"] == backend and int(r["K"]) in KS:
            p = ROOT / "results" / r["run_id"] / "metrics.json"
            if p.exists():
                out[int(r["K"])] = json.loads(p.read_text(encoding="utf-8"))
    return out


def summarise(backend: str, runs: dict[int, dict]) -> list[dict]:
    if not runs:
        return []
    common = set.intersection(*[{w["writer"] for w in rec["writers"]} for rec in runs.values()])
    rows = []
    for K in sorted(runs):
        rec = runs[K]
        ws = rec["writers"]
        cw = [w for w in ws if w["writer"] in common]
        cov = rec.get("target_coverage") or {}
        rows.append({
            "backend": backend, "K": K, "writers_scored": len(ws),
            "targets_scored": cov.get("targets_scored", sum(w["n_targets"] for w in ws)),
            "targets_requested": cov.get("targets_requested", 10 * 20),
            "cer_all_writers": rec["aggregate"]["all_cer"], "hwd_all_writers": rec["aggregate"]["all_hwd"],
            "wid_all_writers": rec["aggregate"]["all_wid_dist"],
            "common_writers": len(common),
            "cer_common": float(np.mean([w["all"]["cer"] for w in cw])),
            "hwd_common": float(np.mean([w["all"]["hwd"] for w in cw])),
            "wid_common": float(np.mean([w["all"]["wid_dist"] for w in cw])),
            "writers_not_scored": sorted(set(rec.get("skipped_writers", []))) or "",
        })
    return rows


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    any_rows = False
    for backend, color in (("baseline", "tab:blue"), ("vatr", "tab:red")):
        rows = summarise(backend, load_runs(backend))
        if not rows:
            continue
        any_rows = True
        with (FIG / f"k_sweep_{backend}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        for ax, suffix, title in ((axes[0], "all_writers", "writers scored at each K (set shrinks with K)"),
                                  (axes[1], "common", f"the {rows[0]['common_writers']} writers scored at every K")):
            x, y = [r[f"cer_{suffix}"] for r in rows], [r[f"hwd_{suffix}"] for r in rows]
            ax.plot(x, y, "-o", color=color, label=backend)
            for r, xi, yi in zip(rows, x, y):
                n = r["writers_scored"] if suffix == "all_writers" else r["common_writers"]
                ax.annotate(f"K={r['K']} (n={n})",
                            (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax.set_xlabel("CER (legibility, lower = better)")
            ax.set_ylabel("HWD (style fidelity, lower = better)")
            ax.set_title(title, fontsize=10)
    if not any_rows:
        print("no sweep rows in the protocol table", file=sys.stderr)
        return 2
    for ax in axes:
        ax.legend()
    fig.suptitle("Legibility vs fidelity over K (20-writer test subset, CPU-reduced)")
    fig.savefig(FIG / "k_sweep.png", dpi=130)
    print("wrote", FIG / "k_sweep.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
