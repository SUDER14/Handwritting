"""Task 1.2 sanity check for the HWD instrument: does it separate writers?

For each of N random pairs of TEST writers (A, B) (fixed seed):  each writer's real lines are split at random into two
DISJOINT halves (A1/A2, B1/B2);  same-writer = HWD(A1, A2),  different-writer = HWD(A1, B2).  A pair "passes" when
same < different. Gate 1 needs >= 90% of pairs to pass. Per-line VGG tokens are cached under data/processed/hwd_cache.

    python scripts/hwd_sanity.py [--pairs 200] [--seed 1337]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/hwd_sanity.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from src.data.iam import load_iam
    from src.evaluation.harness import git_info
    from src.evaluation.hwd_metric import HWDFeatures, hwd, set_mean
    from src.utils.config import resolve_path

    ds = load_iam("test", unit="lines")
    cache = resolve_path("data/processed/hwd_cache")
    cache.mkdir(parents=True, exist_ok=True)
    feats = HWDFeatures()
    by_writer: dict[str, list[int]] = {}
    for i, s in enumerate(ds.samples):
        by_writer.setdefault(s.writer_id, []).append(i)
    by_writer = {w: ix for w, ix in by_writer.items() if len(ix) >= 4}      # need two non-trivial halves

    entry = {}
    todo = [i for ix in by_writer.values() for i in ix]
    for n, i in enumerate(todo):
        sid = ds.samples[i].sample_id
        path = cache / f"{sid}.npz"
        if path.exists():
            z = np.load(path)
            entry[i] = (z["s"], int(z["n"]))
        else:
            s, c = feats.line(ds.read_native(i))
            np.savez(path, s=s, n=c)
            entry[i] = (s, c)
        if (n + 1) % 200 == 0:
            print(f"features {n + 1}/{len(todo)}", flush=True)

    rng = np.random.default_rng(args.seed)
    halves = {}
    for w, ix in by_writer.items():
        p = rng.permutation(ix)
        halves[w] = (set_mean([entry[i] for i in p[: len(p) // 2]]), set_mean([entry[i] for i in p[len(p) // 2:]]))
    writers = sorted(halves)
    same, diff, passed = [], [], 0
    for _ in range(args.pairs):
        a, b = rng.choice(len(writers), 2, replace=False)
        wa, wb = writers[a], writers[b]
        s, d = hwd(halves[wa][0], halves[wa][1]), hwd(halves[wa][0], halves[wb][1])
        same.append(s); diff.append(d); passed += s < d
    res = {"pairs": args.pairs, "writers_used": len(writers), "seed": args.seed, "frac_same_lt_diff": passed / args.pairs,
           "hwd_same_mean": float(np.mean(same)), "hwd_diff_mean": float(np.mean(diff)), "git": git_info()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "git"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
