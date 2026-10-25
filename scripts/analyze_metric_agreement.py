"""ROADMAP Task 5.3 (analysis stage): reads results/metric_agreement/samples.csv (scripts/metric_agreement.py) and reports
  * Spearman rho between writer-ID distance and HWD per sample: overall (real + baseline + glyph_vae), per backend, and
    within-writer (mean of per-writer rhos, which removes writer-level differences);
  * the aggregate pattern (mean wid, mean hwd per backend) -- does glyph_vae beat baseline on writer-ID but not on HWD?
  * explanation tests: how each metric depends on a TEXTURE gap (stroke width, ink density) vs a GEOMETRY gap (lean,
    components per word) to the writer's real lines, and what happens when exactly one of them is repaired
    (interventions: stroke matched / slant matched / both);
  * the 10 samples where the two metrics disagree most (rank-percentile difference) -> montage + csv.
Writes results/metric_agreement/summary.json, results/metric_agreement/disagreement_examples.csv,
results/figures/metric_scatter.png, results/figures/metric_disagreement.png. Nothing here is a claim by itself: the
write-up (docs/metric_agreement.md) quotes these numbers.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

OUT = ROOT / "results" / "metric_agreement"
FIG = ROOT / "results" / "figures"
IMG = ROOT / "data" / "processed" / "agreement"
NUM = ["wid_dist", "hwd", "stroke_px", "lean_deg", "ink_density", "cc_per_word", "real_stroke_px", "real_lean_deg",
       "real_ink_density", "real_cc_per_word", "wid_stroke", "hwd_stroke", "wid_slant", "hwd_slant", "wid_both", "hwd_both",
       "stroke_after_stroke", "lean_after_slant"]


def load() -> list[dict]:
    rows = []
    for r in csv.DictReader((OUT / "samples.csv").open(newline="", encoding="utf-8")):
        for k in NUM:
            r[k] = float(r[k]) if r.get(k, "") != "" else float("nan")
        r["idx"] = int(r["idx"])
        rows.append(r)
    return rows


def rho(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    r, p = spearmanr(x[ok], y[ok])
    return {"rho": round(float(r), 4), "p": float(p), "n": int(ok.sum())}


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def main() -> int:
    rows = load()
    backends = [b for b in ("real", "baseline", "glyph_vae") if any(r["backend"] == b for r in rows)]
    by = {b: [r for r in rows if r["backend"] == b] for b in backends}
    S: dict = {"n_samples": {b: len(by[b]) for b in backends}}

    # 1. agreement of the two metrics
    S["spearman_wid_vs_hwd"] = {"overall": rho(col(rows, "wid_dist"), col(rows, "hwd"))}
    for b in backends:
        S["spearman_wid_vs_hwd"][b] = rho(col(by[b], "wid_dist"), col(by[b], "hwd"))
        per_w = []
        for w in sorted({r["writer"] for r in by[b]}):
            sub = [r for r in by[b] if r["writer"] == w]
            if len(sub) >= 5:
                v = rho(col(sub, "wid_dist"), col(sub, "hwd"))["rho"]
                if v is not None:
                    per_w.append(v)
        S["spearman_wid_vs_hwd"][b]["mean_within_writer_rho"] = round(float(np.mean(per_w)), 4) if per_w else None
        S["spearman_wid_vs_hwd"][b]["n_writers_within"] = len(per_w)

    # 2. aggregate pattern (per-sample means; the protocol's own numbers are set-level)
    S["means"] = {b: {"wid_dist": float(np.nanmean(col(by[b], "wid_dist"))), "hwd": float(np.nanmean(col(by[b], "hwd"))),
                      "stroke_px": float(np.nanmean(col(by[b], "stroke_px"))),
                      "lean_deg": float(np.nanmean(col(by[b], "lean_deg"))),
                      "ink_density": float(np.nanmean(col(by[b], "ink_density"))),
                      "cc_per_word": float(np.nanmean(col(by[b], "cc_per_word")))} for b in backends}

    # 3. gaps to the writer's real lines (generated samples only) and what each metric correlates with
    gen = [r for r in rows if r["backend"] != "real"]
    for r in rows:
        r["gap_stroke"] = abs(r["stroke_px"] - r["real_stroke_px"]) / r["real_stroke_px"]            # texture
        r["gap_density"] = abs(r["ink_density"] - r["real_ink_density"]) / r["real_ink_density"]     # texture
        r["gap_lean"] = abs(r["lean_deg"] - r["real_lean_deg"])                                       # geometry
        r["gap_cc"] = abs(r["cc_per_word"] - r["real_cc_per_word"]) / r["real_cc_per_word"]          # geometry
    gaps = ["gap_stroke", "gap_density", "gap_lean", "gap_cc"]
    S["mean_gap"] = {b: {g: float(np.nanmean(col(by[b], g))) for g in gaps} for b in backends}
    S["gap_correlations_generated"] = {
        g: {"wid_dist": rho(col(gen, g), col(gen, "wid_dist")), "hwd": rho(col(gen, g), col(gen, "hwd"))} for g in gaps}
    for b in ("baseline", "glyph_vae"):
        if b in by:
            S["gap_correlations_" + b] = {
                g: {"wid_dist": rho(col(by[b], g), col(by[b], "wid_dist")), "hwd": rho(col(by[b], g), col(by[b], "hwd"))}
                for g in gaps}

    # 4. interventions: change of each metric when ONE aspect of the generated line is repaired
    S["interventions"] = {}
    for b in ("baseline", "glyph_vae"):
        if b not in by:
            continue
        d = {}
        for tag in ("stroke", "slant", "both"):
            dw = col(by[b], f"wid_{tag}") - col(by[b], "wid_dist")
            dh = col(by[b], f"hwd_{tag}") - col(by[b], "hwd")
            d[tag] = {"mean_delta_wid": float(np.nanmean(dw)), "mean_delta_hwd": float(np.nanmean(dh)),
                      "rel_delta_wid": float(np.nanmean(dw) / np.nanmean(col(by[b], "wid_dist"))),
                      "rel_delta_hwd": float(np.nanmean(dh) / np.nanmean(col(by[b], "hwd"))),
                      "frac_wid_improved": float(np.nanmean(dw < 0)), "frac_hwd_improved": float(np.nanmean(dh < 0)),
                      "n": int(np.isfinite(dw).sum())}
        d["achieved"] = {"mean_abs_stroke_gap_before": float(np.nanmean(np.abs(col(by[b], "stroke_px") - col(by[b], "real_stroke_px")))),
                         "mean_abs_stroke_gap_after": float(np.nanmean(np.abs(col(by[b], "stroke_after_stroke") - col(by[b], "real_stroke_px")))),
                         "mean_abs_lean_gap_before": float(np.nanmean(np.abs(col(by[b], "lean_deg") - col(by[b], "real_lean_deg")))),
                         "mean_abs_lean_gap_after": float(np.nanmean(np.abs(col(by[b], "lean_after_slant") - col(by[b], "real_lean_deg"))))}
        S["interventions"][b] = d

    # 5. disagreement examples: rank percentiles over ALL samples (real + generated)
    pw = rankdata(col(rows, "wid_dist")) / len(rows)
    ph = rankdata(col(rows, "hwd")) / len(rows)
    diff = pw - ph                         # >0: writer-ID says "far", HWD says "near"
    order = np.argsort(diff)
    pick = list(order[::-1][:5]) + list(order[:5])
    ex = []
    for i in pick:
        r = rows[i]
        ex.append({"backend": r["backend"], "writer": r["writer"], "idx": r["idx"], "text": r["text"],
                   "wid_dist": r["wid_dist"], "hwd": r["hwd"], "wid_pct": round(float(pw[i]), 3), "hwd_pct": round(float(ph[i]), 3),
                   "disagreement": round(float(diff[i]), 3),
                   "reading": ("writer-ID: far / HWD: near" if diff[i] > 0 else "writer-ID: near / HWD: far")})
    with (OUT / "disagreement_examples.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ex[0]))
        w.writeheader()
        w.writerows(ex)
    S["disagreement_examples"] = ex
    S["spearman_of_percentiles_check"] = rho(pw, ph)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for b, c in (("real", "tab:green"), ("baseline", "tab:blue"), ("glyph_vae", "tab:orange")):
        if b in by:
            ax.scatter(col(by[b], "wid_dist"), col(by[b], "hwd"), s=14, alpha=0.6, color=c, label=b)
    ax.set_xlabel("writer-ID cosine distance (per sample)")
    ax.set_ylabel("HWD (per sample)")
    o = S["spearman_wid_vs_hwd"]["overall"]
    ax.set_title(f"Per-sample metric agreement: overall Spearman rho {o['rho']}", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "metric_scatter.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(ex), 1, figsize=(11, 2.0 * len(ex)))
    for ax, e in zip(np.atleast_1d(axes), ex):
        sub = e["backend"]
        paths = [IMG / sub / f"{e['writer']}_{e['idx']:02d}.png", IMG / "ref" / f"{e['writer']}.png"]
        imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
        if any(im is None for im in imgs):
            ax.text(0.5, 0.5, "image missing", ha="center")
            ax.axis("off")
            continue
        W = max(im.shape[1] for im in imgs)
        canvas = np.full((sum(im.shape[0] for im in imgs) + 6, W), 255, np.uint8)
        y = 0
        for im in imgs:
            canvas[y:y + im.shape[0], :im.shape[1]] = im
            y += im.shape[0] + 6
        ax.imshow(canvas, cmap="gray", aspect="equal")
        ax.set_title(f"{e['backend']} w{e['writer']} #{e['idx']} (top: this line, bottom: writer's reference) | "
                     f"wid {e['wid_dist']:.3f} (pct {e['wid_pct']}), HWD {e['hwd']:.2f} (pct {e['hwd_pct']}) -> {e['reading']}",
                     fontsize=7.5, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "metric_disagreement.png", dpi=120)
    plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(S, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: S[k] for k in ("n_samples", "spearman_wid_vs_hwd", "means", "mean_gap", "interventions")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
