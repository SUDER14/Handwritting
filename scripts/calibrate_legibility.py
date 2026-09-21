"""Calibration: what CER does the CNN legibility scorer get on REAL human handwriting? (the floor)

    python scripts/calibrate_legibility.py                 # 200 held-out crops from the TEST split writers
    python scripts/calibrate_legibility.py --split val --unit words

Run this before reading any generation result. If real handwriting already scores a high CER, the scorer
cannot resolve differences between generators. Writes results/<utc>_<sha>_calibration/calibration.json.
The test split is used by default because these are crops from writers no model was trained or tuned on;
nothing here is fit to them, so this does not consume the test split's held-out status for the generators.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import SPLIT_NAMES, load_iam
from src.evaluation import calibration, harness
from src.recognition.cnn_recognizer import CNNRecognizer
from src.utils.config import load_config, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=SPLIT_NAMES, default="test")
    ap.add_argument("--unit", choices=["lines", "words"], default="lines")
    ap.add_argument("--n", type=int, default=None, help="default: writer_id.calibration.n_crops (200)")
    ap.add_argument("--iam-root", default=None)
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    cal = cfg["writer_id"]["calibration"]
    n = args.n or cal["n_crops"]
    cnn_path = resolve_path(cfg["recognition"]["cnn_checkpoint_path"])
    try:
        recognizer = CNNRecognizer(checkpoint_path=str(cnn_path))
        dataset = load_iam(args.split, unit=args.unit, root=args.iam_root, splits_dir=args.splits_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    records, agg = calibration.run_calibration(dataset, recognizer, cfg, n, cal["seed"])
    git = harness.git_info()
    stamp = datetime.now(timezone.utc)
    results_dir = Path(args.results_dir) if args.results_dir else resolve_path(cfg["evaluation"]["results_dir"])
    run_dir = results_dir / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{git['sha']}{'-dirty' if git['dirty'] else ''}_calibration"
    run_dir.mkdir(parents=True, exist_ok=False)
    out = {"kind": "legibility_calibration", "timestamp_utc": stamp.isoformat(timespec="seconds"), "git": git,
           "split": args.split, "unit": args.unit, "cli_args": vars(args), "resolved_config": copy.deepcopy(cfg),
           "cnn_checkpoint": harness.checkpoint_record(cnn_path), "aggregate": agg, "samples": records}
    (run_dir / "calibration.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"\nCNN legibility scorer on REAL handwriting: {args.split} split, unit={args.unit}, "
          f"{agg['n_scored']}/{agg['n_requested']} crops, {agg['n_writers']} writers, {agg['n_reference_letters']} reference letters")
    print(f"  CER micro {agg['cer_micro']:.3f} | mean {agg['cer_mean']:.3f} | median {agg['cer_median']:.3f} "
          f"| p25-p75 {agg['cer_p25']:.3f}-{agg['cer_p75']:.3f} | p90 {agg['cer_p90']:.3f}")
    print(f"  null decoders (same glyph count): random letters {agg['null_cer_random_letters']:.3f}, "
          f"constant 'e' {agg['null_cer_constant_e']:.3f}")
    print(f"  segmentation: glyph count == letter count for {agg['glyph_count_match_rate']:.1%} of crops "
          f"({agg['glyphs_per_reference_letter']:.2f} glyphs per reference letter)")
    if agg["n_failed"]:
        print(f"  {agg['n_failed']} crops failed to score (see the json)")
    if agg["cer_micro"] >= 0.6:
        print("\n  !!! FLOOR IS HIGH: real handwriting scores CER >= 0.6. This scorer cannot resolve differences between\n"
              "  !!! generators; do not read generation results through it until the recognizer/segmentation improves.")
    print(f"\nwrote {run_dir / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
