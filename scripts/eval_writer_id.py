"""Evaluate a trained writer-ID run by retrieval on a split's writers (cosine distance): top-1, mAP, chance.

    python scripts/eval_writer_id.py --run models/checkpoints/<run_id> --split val
    python scripts/eval_writer_id.py --run models/checkpoints/<run_id> --split val --mode full_line
    python scripts/eval_writer_id.py --run models/checkpoints/<run_id> --split test --allow-test

Compare embedding modes / gallery protocols on VAL; the test split needs --allow-test (training already reports
it once). Results are written next to the run as eval_<split>_<mode>_<utc>.json (new file each time).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import SPLIT_NAMES, load_iam_splits
from src.utils.config import load_config, resolve_path
from src.writer_id.data import LineStore
from src.writer_id.retrieval import embed_samples, retrieval_metrics
from src.writer_id.runs import best_checkpoint, load_encoder, write_once


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory under models/checkpoints/")
    ap.add_argument("--split", choices=SPLIT_NAMES, default="val")
    ap.add_argument("--mode", choices=["window_mean", "full_line"], default=None)
    ap.add_argument("--include-same-form", action="store_true", help="optimistic protocol: only exclude the query itself")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--iam-root", default=None)
    ap.add_argument("--splits-dir", default=None)
    args = ap.parse_args()
    if args.split == "test" and not args.allow_test:
        print("Refusing to evaluate the test split without --allow-test.", file=sys.stderr)
        return 2

    run_dir = Path(args.run)
    encoder, run_cfg = load_encoder(run_dir)
    wid = copy.deepcopy(run_cfg["writer_id"])
    if args.mode:
        wid["eval"]["mode"] = args.mode
    if args.include_same_form:
        wid["eval"]["exclude_same_form"] = False

    cfg = load_config()
    try:
        ds = load_iam_splits(unit="lines", config=cfg, root=args.iam_root, splits_dir=args.splits_dir)[args.split]
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    store = LineStore(resolve_path(cfg["writer_id"]["cache_dir"]), wid["input_height"], wid["binarize"])
    store.ensure(ds)
    emb = embed_samples(encoder, store, ds.samples, wid["eval"])
    m = retrieval_metrics(emb, [s.writer_id for s in ds.samples], [s.form_id for s in ds.samples],
                          exclude_same_form=wid["eval"]["exclude_same_form"])
    m.update({"split": args.split, "mode": wid["eval"]["mode"], "checkpoint": best_checkpoint(run_dir).name})
    out = run_dir / f"eval_{args.split}_{m['mode']}_{'sameform' if not m['exclude_same_form'] else 'crossform'}_" \
                    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_once(out, json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))
    thr = wid["eval"]["min_usable_top1"]
    print(f"top-1 {m['top1']:.3f} vs chance {m['chance_top1']:.3f}; usability threshold {thr}: "
          f"{'PASS' if m['top1'] >= thr else 'BELOW THRESHOLD -- not a usable writer-identity metric'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
