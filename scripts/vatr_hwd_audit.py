"""Task 3.2 protocol audit: why does VATr's reproduced HWD (0.6069) differ from the published 0.828?

Works on the images vatr_reproduce.py already generated (<images>/Fake_test, <images>/Real_test); nothing is regenerated.
VGG features are computed ONCE per image (token-sum + token-count, batch size 1 as HWD requires) for each preprocessing
variant; every score below is then an aggregation of those cached sums, so subsets, floors and variants are cheap.

Preprocessing variants (all through the pinned HWD VGG16 backbone, third_party/HWD @ eabb5b5):
  pinned     PaddingSquareHeight -> ResizeHeight(32)                (what hwd.scores.HWDScore(32) does at eabb5b5)
  era2023    ResizeHeight(32) -> ToTensor -> PaddingMin(32,32)      (what HWDScore did at 6a15ffd, Sept 2023)
  cropwhite  trailing white columns removed, then `pinned`          (undoes VATr's batch-wise right padding)
Aggregations:
  tok_writer   mean over ALL tokens of a writer, distance, averaged over writers      (pinned EuclideanDistance; the
               paper's text: "all the vectors from all the real images sets are gathered and then averaged")
  tok_pooled   same, one pool over the whole set
  img_writer   mean of per-image mean features (global average pool, as at 6a15ffd), averaged over writers
  img_pooled   same, one pool (6a15ffd HWDScore()(a, b))
Cross-checks (the derived numbers must equal the real library code on the same pairs): the pinned HWDScore(32) and the
actual 6a15ffd HWDScore, the latter run in a subprocess from a git worktree.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HWD_REPO = ROOT / "third_party" / "HWD"
PAPER_ERA_SHA = "6a15ffd"          # 2023-09-08, the last commit before the BMVC 2023 camera-ready / arXiv (2023-10-31)
PUBLISHED_HWD = 0.828
VARIANTS = ("pinned", "era2023", "cropwhite")
MODES = ("tok_writer", "tok_pooled", "img_writer", "img_pooled")


# ---------------------------------------------------------------------------------------------------------- features
def list_pairs(img_dir: Path) -> list[tuple[str, Path, Path]]:
    fake, real = img_dir / "Fake_test", img_dir / "Real_test"
    rels = sorted(p.relative_to(fake).as_posix() for p in fake.rglob("*.png"))
    missing = [r for r in rels if not (real / r).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} fakes without a real twin, e.g. {missing[:3]}")
    return [(r.split("/")[0], fake / r, real / r) for r in rels]


def crop_white(img: Image.Image) -> Image.Image:
    cols = np.array(img.convert("L")).mean(0)
    ink = np.nonzero(cols < 250)[0]
    return img.crop((0, 0, int(ink.max()) + 1 if len(ink) else 1, img.size[1]))


def make_transform(variant: str):
    sys.path.insert(0, str(HWD_REPO))
    from hwd.datasets.transforms import Compose, PaddingMin, PaddingSquareHeight, ResizeHeight, ToTensor
    if variant == "era2023":
        return Compose([ResizeHeight(32), ToTensor(), PaddingMin(32, 32)])
    base = Compose([PaddingSquareHeight(), ResizeHeight(32), ToTensor()])
    return (lambda im: base(crop_white(im))) if variant == "cropwhite" else base


@torch.inference_mode()
def token_sums(paths: list[Path], variant: str, model, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per image: sum of VGG token features (float64), token count, original width. Batch size 1 (no padding)."""
    tf = make_transform(variant)
    sums = np.zeros((len(paths), 512), np.float64)
    ntok = np.zeros(len(paths), np.int64)
    widths = np.zeros(len(paths), np.int64)
    t0 = time.time()
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        widths[i] = img.size[0]
        x = tf(img).unsqueeze(0).to(device)
        f = model(x)[0].double().cpu().numpy()                    # (tokens, 512)
        sums[i], ntok[i] = f.sum(0), len(f)
        if i % 5000 == 0:
            print(f"  [{variant}] {i}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    return sums, ntok, widths


# ---------------------------------------------------------------------------------------------------- aggregations
def _dist(sa, na, sb, nb, mode: str) -> float:
    if mode.startswith("tok"):
        return float(np.linalg.norm(sa.sum(0) / na.sum() - sb.sum(0) / nb.sum()))
    return float(np.linalg.norm((sa / na[:, None]).mean(0) - (sb / nb[:, None]).mean(0)))


def score(A, B, mode: str) -> float:
    """A, B = (sums, ntok, writer ids). *_writer: average over the writers of A of the per-writer distance."""
    (sa, na, wa), (sb, nb, wb) = A, B
    if mode.endswith("pooled"):
        return _dist(sa, na, sb, nb, mode)
    vals = []
    for w in np.unique(wa):
        ma, mb = wa == w, wb == w
        if mb.any():
            vals.append(_dist(sa[ma], na[ma], sb[mb], nb[mb], mode))
    return float(np.mean(vals))


def take(F, idx):
    return F[0][idx], F[1][idx], F[2][idx]


# ------------------------------------------------------------------------------------------------------- cross-checks
def copy_subset(pairs, idx, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for i in idx:
        w, f, r = pairs[i]
        for sub, src in (("Fake_test", f), ("Real_test", r)):
            out = dest / sub / w / src.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)


def run_pinned_library(sub: Path) -> float:
    sys.path.insert(0, str(HWD_REPO))
    from hwd.datasets import FolderDataset
    from hwd.scores import HWDScore
    return float(HWDScore(height=32)(FolderDataset(str(sub / "Fake_test")), FolderDataset(str(sub / "Real_test"))))


def run_paper_era_library(sub: Path, work: Path) -> dict:
    """The REAL 6a15ffd HWDScore, from a git worktree, in a subprocess (its package names `metrics`/`datasets` clash with
    the HuggingFace `datasets` package). Reports pooled HWDScore()(fake, real) -- how its README calls it."""
    wt = work / "hwd_paper_era"
    if not wt.exists():
        subprocess.run(["git", "-C", str(HWD_REPO), "fetch", "--unshallow", "--tags"], check=False,
                       capture_output=True)
        subprocess.run(["git", "-C", str(HWD_REPO), "worktree", "add", "--detach", str(wt), PAPER_ERA_SHA], check=True)
    runner = ROOT / "scripts" / "_hwd_paper_era_run.py"
    out = subprocess.run([sys.executable, str(runner), str(wt), str(sub / "Fake_test"), str(sub / "Real_test")],
                         capture_output=True, text=True)
    if out.returncode:
        return {"error": (out.stderr or out.stdout)[-1500:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


# -------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data" / "processed" / "vatr_repro" / "vatr"))
    ap.add_argument("--out", default=str(ROOT / "results" / "vatr_hwd_audit.json"))
    ap.add_argument("--work", default=str(ROOT / "data" / "processed" / "hwd_audit"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-library-checks", action="store_true")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    img_dir, work = Path(args.images), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pairs = list_pairs(img_dir)
    n = len(pairs)
    writers = sorted({p[0] for p in pairs})
    wid = np.array([writers.index(p[0]) for p in pairs])
    print(f"{n} pairs, {len(writers)} writers", flush=True)

    sys.path.insert(0, str(HWD_REPO))
    from hwd.scores import HWDScore
    model = HWDScore(height=32).backbone.model.to(args.device).eval()

    res: dict = {"n_pairs": n, "n_writers": len(writers), "published": PUBLISHED_HWD, "seed": args.seed,
                 "paper_era_sha": PAPER_ERA_SHA, "variants": {}}
    sub_idx = np.sort(rng.choice(n, size=min(args.subset, n), replace=False))
    perm = np.arange(len(writers))
    shift = (perm + 1) % len(writers)                               # derangement: writer w -> writer w+1 (sorted)

    for v in args.variants:
        sf, nf, wf = token_sums([p[1] for p in pairs], v, model, args.device)
        sr, nr, wr = token_sums([p[2] for p in pairs], v, model, args.device)
        F, R = (sf, nf, wid), (sr, nr, wid)
        entry: dict = {"mean_width_fake": float(wf.mean()), "mean_width_real": float(wr.mean()),
                       "mean_tokens_fake": float(nf.mean()), "mean_tokens_real": float(nr.mean())}
        entry["full"] = {m: score(F, R, m) for m in MODES}
        entry["subset"] = {m: score(take(F, sub_idx), take(R, sub_idx), m) for m in MODES}

        # (c) floors, same data and aggregation. Real halves are disjoint; halves hold n/2 images per writer, so the
        # floor carries a small-sample inflation that the full-size comparison does not.
        a_idx, b_idx = [], []
        for w in range(len(writers)):
            ii = rng.permutation(np.nonzero(wid == w)[0])
            a_idx.append(ii[: len(ii) // 2])
            b_idx.append(ii[len(ii) // 2: 2 * (len(ii) // 2)])
        a_idx, b_idx = np.concatenate(a_idx), np.concatenate(b_idx)
        entry["same_writer_floor_real_half_vs_real_half"] = {m: score(take(R, a_idx), take(R, b_idx), m) for m in MODES}
        # fake vs the OTHER half of the same writer's reals (fakes' own twins excluded), same sample size
        entry["fake_half_vs_real_other_half"] = {m: score(take(F, a_idx), take(R, b_idx), m) for m in MODES}
        # different writer: writer w's reals against writer (w+1)'s reals (all images, full size)
        R_shift = (sr, nr, shift[wid])
        entry["diff_writer_real_vs_next_writer_real"] = {m: score(R, R_shift, m) for m in MODES}
        entry["diff_writer_fake_vs_next_writer_real"] = {m: score(F, R_shift, m) for m in MODES}

        # (d) sampling noise of the full score over writers (what a different style-reference draw can move)
        per_writer = np.array([_dist(sf[wid == w], nf[wid == w], sr[wid == w], nr[wid == w], "tok_writer")
                               for w in range(len(writers))])
        boots = [per_writer[rng.integers(0, len(per_writer), len(per_writer))].mean() for _ in range(2000)]
        entry["per_writer_tok"] = {"mean": float(per_writer.mean()), "std": float(per_writer.std(ddof=1)),
                                   "se": float(per_writer.std(ddof=1) / np.sqrt(len(per_writer))),
                                   "boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        res["variants"][v] = entry
        print(json.dumps({v: entry}, indent=2), flush=True)

    if not args.skip_library_checks:
        sub = work / "subset"
        copy_subset(pairs, sub_idx, sub)
        lib = {"pinned_HWDScore32_on_subset": run_pinned_library(sub),
               "pinned_derived_tok_writer_on_subset": res["variants"].get("pinned", {}).get("subset", {}).get("tok_writer"),
               "paper_era_6a15ffd": run_paper_era_library(sub, work),
               "era2023_derived_img_pooled_on_subset": res["variants"].get("era2023", {}).get("subset", {}).get("img_pooled")}
        res["library_checks"] = lib
        print(json.dumps(lib, indent=2), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
