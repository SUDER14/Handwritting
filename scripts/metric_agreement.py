"""ROADMAP Task 5.3 (data stage): per-SAMPLE writer-ID distance and HWD for real / baseline / glyph_vae lines, plus controlled
interventions that test WHY the two style metrics disagree. The analysis (Spearman, disagreement examples, figures) is
scripts/analyze_metric_agreement.py; this script only produces results/metric_agreement/samples.csv (+ images under
data/processed/agreement/, gitignored). Heavy (generation + torch): run it detached, nothing else heavy alongside.

Same instruments as the protocol (pinned CRNN-free: only the writer-ID net and HWD are needed here), same 20-writer
seed-1337 subset, K=1, same seeded reference/target split (`protocol.split_lines`). Per generated line g of writer w:
    wid_dist = 1 - cos(embed(g), mean unit embedding of w's real held-out target lines)
    hwd      = || mean token(g) - mean token(w's real held-out target lines) ||   (official HWD backbone, see hwd_metric.py)
For REAL lines the same two numbers are computed leave-one-out (the line is removed from its writer's real set).
Descriptors per line (native ink mask): stroke width, lean, ink density, connected components per word.
Interventions on every generated line (texture vs geometry):
    stroke : dilate/erode until the line's stroke width matches the writer's real median stroke width  (texture)
    slant  : shear until the line's lean matches the writer's real median lean                          (geometry)
    both   : stroke then slant
and the two metrics are recomputed on the modified line.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.writer_features import binarize_ink, components_per_word, lean_deg, stroke_width_px  # noqa: E402
from src.data.iam import load_iam  # noqa: E402
from src.evaluation import protocol as P  # noqa: E402
from src.evaluation.harness import render_text_clean, truncate_text  # noqa: E402
from src.evaluation.hwd_metric import HWDFeatures, hwd, set_mean  # noqa: E402
from src.evaluation.writer_id import get_writer_embedder  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

logger = get_logger(__name__)
OUT = ROOT / "results" / "metric_agreement"
IMG = ROOT / "data" / "processed" / "agreement"
COLUMNS = ["backend", "writer", "idx", "text", "wid_dist", "hwd", "stroke_px", "lean_deg", "ink_density", "cc_per_word",
           "real_stroke_px", "real_lean_deg", "real_ink_density", "real_cc_per_word",
           "wid_stroke", "hwd_stroke", "wid_slant", "hwd_slant", "wid_both", "hwd_both",
           "stroke_after_stroke", "lean_after_slant"]


def ink_density(ink: np.ndarray) -> float:
    return float((np.asarray(ink) > 127).mean())


def match_stroke(ink: np.ndarray, target_px: float, max_steps: int = 8) -> np.ndarray:
    """Dilate / erode (3x3 ellipse, one step at a time) until the measured stroke width is within 10% of the target."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cur = (np.asarray(ink) > 127).astype(np.uint8) * 255
    best, best_err = cur, abs(stroke_width_px(cur) - target_px)
    for _ in range(max_steps):
        s = stroke_width_px(cur)
        if abs(s - target_px) <= 0.1 * target_px:
            return cur
        cur = cv2.dilate(cur, k) if s < target_px else cv2.erode(cur, k)
        if not cur.any():
            break
        err = abs(stroke_width_px(cur) - target_px)
        if err < best_err:
            best, best_err = cur, err
    return best


def shear_to(ink: np.ndarray, delta_deg: float) -> np.ndarray:
    """Lean the writing by `delta_deg` more to the right (negative = left): x' = x + tan(d) * (h - y), nearest-neighbour."""
    if abs(delta_deg) < 0.5:
        return ink
    h = ink.shape[0]
    t = np.tan(np.radians(delta_deg))
    pad = int(abs(t) * h) + 2
    p = cv2.copyMakeBorder(np.asarray(ink, np.uint8), 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    m = np.array([[1, t, -t * h], [0, 1, 0]], np.float32)         # WARP_INVERSE_MAP: x_src = x_dst - t*(h - y)
    return cv2.warpAffine(p, m, (p.shape[1], p.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def to_gray(ink: np.ndarray) -> np.ndarray:
    return (255 - (np.asarray(ink) > 127).astype(np.uint8) * 255).astype(np.uint8)


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    ev, pc = cfg["evaluation"], cfg["evaluation"]["protocol"]
    embedder = get_writer_embedder("trained", cfg, run_dir=None)
    assert embedder.usable and not embedder.is_stub, "writer-ID instrument must be the pinned, usable one"
    hf = HWDFeatures()
    scorer = P.Scorer(None, embedder, hf, cfg["crnn"]["input_height"])
    ds = load_iam("test", unit="lines")
    by: dict[str, list] = {}
    for s in ds.samples:
        by.setdefault(s.writer_id, []).append(s)
    index = {s.sample_id: i for i, s in enumerate(ds.samples)}
    writers = P.choose_writers(ds, pc["n_writers"], pc["seed"], max(pc["ks"]) + pc["targets_per_writer"] // 2 + 1)
    pcfg = {"targets_per_writer": pc["targets_per_writer"], "min_reference_letters": ev["min_reference_letters"],
            "max_target_chars": ev["max_target_chars"]}
    K = 1
    from src.evaluation import backends as B
    backends = {"baseline": B.BaselineBackend(cfg),
                "glyph_vae": B.NeuralBackend(cfg, str(resolve_path(cfg["training"]["checkpoint_path"])))}
    OUT.mkdir(parents=True, exist_ok=True)
    rows, done_writers = [], set()
    if (OUT / "samples.csv").exists():                  # resume: keep the finished writers of an interrupted run
        rows = list(csv.DictReader((OUT / "samples.csv").open(newline="", encoding="utf-8")))
        done_writers = {r["writer"] for r in rows}
        logger.info("resuming: %d rows, writers done %s", len(rows), sorted(done_writers))

    def metrics_vs_set(g, real_emb_mean, real_h_mean):
        if not scorer.scorable(g):               # an intervention (erosion) can erase a thin line: no value, not a crash
            return float("nan"), float("nan")
        e = scorer.embed(g)
        e = e / np.linalg.norm(e)
        wid = float(1.0 - np.dot(e, real_emb_mean) / np.linalg.norm(real_emb_mean))
        return wid, hwd(set_mean([scorer.hwd_entry(g)]), real_h_mean)

    for wi, w in enumerate(writers):
        if w in done_writers:
            continue
        sp = P.split_lines(by[w], K, pcfg["targets_per_writer"], pcfg["min_reference_letters"], pc["seed"], w)
        refs, targets = sp
        real = [ds.read_native(index[t.sample_id]) for t in targets]
        texts = [truncate_text(t.transcription, pcfg["max_target_chars"]) for t in targets]
        r_emb = [scorer.embed(g) for g in real]
        r_unit = [e / np.linalg.norm(e) for e in r_emb]
        r_h = [scorer.hwd_entry(g, t.sample_id) for g, t in zip(real, targets)]
        r_inks = [binarize_ink(g) for g in real]
        r_str = [stroke_width_px(i) for i in r_inks]
        r_lean = [lean_deg(i) for i in r_inks]
        r_den = [ink_density(i) for i in r_inks]
        r_cc = [components_per_word(i, max(len(t.split()), 1)) for i, t in zip(r_inks, texts)]
        w_str, w_lean = float(np.median(r_str)), float(np.median(r_lean))
        w_den, w_cc = float(np.mean(r_den)), float(np.mean([c for c in r_cc if c is not None]))
        for sub, imgs in (("real", real),):
            (IMG / sub).mkdir(parents=True, exist_ok=True)
            for i, g in enumerate(imgs):
                cv2.imwrite(str(IMG / sub / f"{w}_{i:02d}.png"), g)
        (IMG / "ref").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(IMG / "ref" / f"{w}.png"), ds.read_native(index[refs[0].sample_id]))
        # REAL rows: leave-one-out against the writer's other real target lines
        for i in range(len(real)):
            others = [j for j in range(len(real)) if j != i]
            if not others:
                continue
            mean_e = np.mean([r_unit[j] for j in others], axis=0)
            wid = float(1.0 - np.dot(r_unit[i], mean_e) / np.linalg.norm(mean_e))
            rows.append({"backend": "real", "writer": w, "idx": i, "text": texts[i], "wid_dist": wid,
                         "hwd": hwd(set_mean([r_h[i]]), set_mean([r_h[j] for j in others])), "stroke_px": r_str[i],
                         "lean_deg": r_lean[i], "ink_density": r_den[i], "cc_per_word": r_cc[i],
                         "real_stroke_px": w_str, "real_lean_deg": w_lean, "real_ink_density": w_den,
                         "real_cc_per_word": w_cc})
        mean_e_all = np.mean(r_unit, axis=0)
        mean_h_all = set_mean(r_h)
        ref_objs = [P.Ref(ds.read_native(index[r.sample_id]), r.transcription) for r in refs]
        for name, be in backends.items():
            for i, t in enumerate(texts):
                try:
                    g = be.generate(ref_objs, t) if t.strip() else None
                except Exception as e:                                   # noqa: BLE001
                    logger.warning("%s writer %s target %d failed: %r", name, w, i, e)
                    g = None
                if g is None or not np.any(np.asarray(g) < 128) or not scorer.scorable(g):
                    logger.warning("%s writer %s target %d: no scorable output, skipped", name, w, i)
                    continue
                (IMG / name).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(IMG / name / f"{w}_{i:02d}.png"), g)
                wid, h = metrics_vs_set(g, mean_e_all, mean_h_all)
                ink = binarize_ink(g)
                row = {"backend": name, "writer": w, "idx": i, "text": t, "wid_dist": wid, "hwd": h,
                       "stroke_px": stroke_width_px(ink), "lean_deg": lean_deg(ink), "ink_density": ink_density(ink),
                       "cc_per_word": components_per_word(ink, max(len(t.split()), 1)), "real_stroke_px": w_str,
                       "real_lean_deg": w_lean, "real_ink_density": w_den, "real_cc_per_word": w_cc}
                s_ink = match_stroke(ink, w_str)
                row["stroke_after_stroke"] = stroke_width_px(s_ink)
                row["wid_stroke"], row["hwd_stroke"] = metrics_vs_set(to_gray(s_ink), mean_e_all, mean_h_all)
                l_ink = shear_to(ink, w_lean - row["lean_deg"])
                row["lean_after_slant"] = lean_deg(l_ink)
                row["wid_slant"], row["hwd_slant"] = metrics_vs_set(to_gray(l_ink), mean_e_all, mean_h_all)
                b_ink = shear_to(s_ink, w_lean - lean_deg(s_ink))
                row["wid_both"], row["hwd_both"] = metrics_vs_set(to_gray(b_ink), mean_e_all, mean_h_all)
                rows.append(row)
        logger.info("writer %s (%d/%d) done, %d rows, %.0fs", w, wi + 1, len(writers), len(rows), time.time() - t0)
        with (OUT / "samples.csv").open("w", newline="", encoding="utf-8") as f:       # incremental: a reaped run keeps its rows
            wr = csv.DictWriter(f, fieldnames=COLUMNS)
            wr.writeheader()
            wr.writerows(rows)
    (OUT / "meta.json").write_text(json.dumps(
        {"writers": writers, "K": K, "n_rows": len(rows), "wall_s": time.time() - t0,
         "writer_id": embedder.describe(), "note": "per-sample metrics; real rows are leave-one-out"}, indent=2) + "\n",
        encoding="utf-8")
    print(f"wrote {len(rows)} rows to {OUT / 'samples.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
