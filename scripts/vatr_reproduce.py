"""ROADMAP Task 3.2: reproduce VATr's published IAM HWD with the released checkpoint and VATr's OWN evaluation code.

Published target: HWD paper (Pippi et al., BMVC 2023), Table 2, VATr on the IAM test set: HWD = 0.828.

Procedure (nothing in third_party/VATr is edited):
  1. run the official `third_party/VATr/generate_fakes.py` (released `files/vatr.pth`, `files/IAM-32.pickle`) via runpy,
     which writes Real_test/<writer>/*.png and Fake_test/<writer>/*.png (one fake per real test word, same text, style
     from 15 random words of the same writer);
  2. official HWD: HWDScore(height=32)(FolderDataset(Fake_test), FolderDataset(Real_test)).

Runtime shims (logged in ROADMAP_STATE.md), needed only because this host has no CUDA / no extra downloads:
  * --device cpu and torch.load(map_location="cpu");
  * `wandb` stubbed (generate_fakes imports train.py, which imports wandb; logging only);
  * VATr builds an FID InceptionV3 in __init__ that generation never calls; its weights are a download outside the
    ROADMAP allow-list, so it is replaced by an empty module;
  * the TRAIN-split fake generation (not needed for the test HWD, ~hours on CPU) is skipped. Consequence: the numpy RNG
    that draws each word's 15 style references starts at seed 742 for the test split instead of after the train pass,
    so the style draws differ from an original full run (the paper's own draw is not published either).
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VATR = ROOT / "third_party" / "VATr"
HWD_REPO = ROOT / "third_party" / "HWD"
PUBLISHED_HWD = 0.828


def generate(out_dir: Path, batch_size: int) -> None:
    import os

    import torch
    import torch.nn as nn

    sys.modules.setdefault("wandb", types.ModuleType("wandb"))
    sys.path.insert(0, str(VATR))
    os.chdir(VATR)

    import models.inception as inc

    class _NoInception(nn.Module):
        BLOCK_INDEX_BY_DIM = inc.InceptionV3.BLOCK_INDEX_BY_DIM

        def __init__(self, *a, **k):
            super().__init__()

    inc.InceptionV3 = _NoInception

    _load = torch.load
    torch.load = lambda f, *a, **k: _load(f, *a, **{"map_location": "cpu", "weights_only": False, **k})

    import models.model as mm
    orig = mm.VATr.save_images_for_fid_calculation

    def test_only(self, path, loader, split="train"):
        if split == "train":
            print("[vatr_reproduce] skipping TRAIN split generation (not needed for the test HWD)")
            return
        return orig(self, path, loader, split)

    mm.VATr.save_images_for_fid_calculation = test_only
    sys.argv = ["generate_fakes.py", "--checkpoint", "files/vatr.pth", "--output", str(out_dir),
                "--device", "cpu", "--batch_size", str(batch_size)]
    runpy.run_path(str(VATR / "generate_fakes.py"), run_name="__main__")


def score(img_dir: Path) -> dict:
    sys.path.insert(0, str(HWD_REPO))
    from hwd.datasets import FolderDataset
    from hwd.scores import HWDScore
    fakes, reals = FolderDataset(str(img_dir / "Fake_test")), FolderDataset(str(img_dir / "Real_test"))
    value = float(HWDScore(height=32)(fakes, reals))
    n_fake = sum(1 for _ in (img_dir / "Fake_test").rglob("*.png"))
    n_real = sum(1 for _ in (img_dir / "Real_test").rglob("*.png"))
    n_writers = sum(1 for p in (img_dir / "Fake_test").iterdir() if p.is_dir())
    return {"hwd": value, "n_fake": n_fake, "n_real": n_real, "n_writers": n_writers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "vatr_repro"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stage", choices=["all", "generate", "score"], default="all")
    args = ap.parse_args()
    out = Path(args.out)
    img_dir = out / "vatr"                                  # generate_fakes appends the checkpoint stem
    t0 = time.time()
    if args.stage in ("all", "generate"):
        generate(out, args.batch_size)
    if args.stage in ("all", "score"):
        from src.evaluation.harness import git_info       # noqa: E402
        res = score(img_dir)
        rel = abs(res["hwd"] - PUBLISHED_HWD) / PUBLISHED_HWD
        rec = {"task": "3.2", "utc": datetime.now(timezone.utc).isoformat(), "git": git_info(),
               "published_hwd": PUBLISHED_HWD, "published_source": "Pippi et al., BMVC 2023 (HWD), Table 2, VATr, IAM test",
               **res, "rel_diff": rel, "within_10pct": rel <= 0.10, "images": str(img_dir),
               "device": "cpu", "shims": ["cpu map_location", "wandb stub", "FID InceptionV3 stub (unused)",
                                          "train-split generation skipped (RNG draw differs)"],
               "wall_s": time.time() - t0}
        dest = ROOT / "results" / "vatr_reproduction.json"
        dest.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
