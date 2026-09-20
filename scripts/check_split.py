"""Audit the IAM writer-disjoint split: assert zero writer overlap, print counts.

Usage:
    python scripts/check_split.py                 # uses data.iam.* from config/config.yaml
    python scripts/check_split.py --root D:/IAM   # IAM lives elsewhere
    python scripts/check_split.py --regenerate    # rewrite data/splits/*.json (deliberate changes only)

Exit code 0 = every check passed, 1 = a check failed, 2 = IAM data not found.
Creates data/splits/{train,val,test}.json on first run; on later runs it
verifies the files on disk against the seed in config.yaml rather than trusting them.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import SPLIT_NAMES, UNITS, IAMCorpus, get_writer_splits, make_writer_splits, read_split_files
from src.utils.config import load_config, resolve_path


def check_unit(unit: str, root, splits_dir: Path, cfg: dict, regenerate: bool) -> list[str]:
    """Run all checks for one unit; returns a list of failure messages (empty = pass)."""
    failures: list[str] = []
    corpus = IAMCorpus(root=root, unit=unit)
    seed, fractions = cfg["split_seed"], cfg["split_fractions"]
    splits = get_writer_splits(corpus.all_writer_ids, seed, fractions, splits_dir, regenerate=regenerate)

    print(f"\n== unit: {unit}  (root={corpus.root}) ==")
    print(f"writers in forms.txt: {len(corpus.all_writer_ids)}   samples indexed: {corpus.num_samples}   "
          f"skipped: {corpus.skipped}")
    print(f"{'split':<7}{'writers':>9}{'w/ samples':>12}{'samples':>10}")

    sample_ids: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        writers = splits[name]
        with_samples = [w for w in writers if corpus.samples_by_writer.get(w)]
        samples = [s for w in writers for s in corpus.samples_by_writer.get(w, [])]
        sample_ids[name] = {s.sample_id for s in samples}
        print(f"{name:<7}{len(writers):>9}{len(with_samples):>12}{len(samples):>10}")
        bad = [s.sample_id for s in samples if s.writer_id not in set(writers)]
        if bad:
            failures.append(f"[{unit}/{name}] {len(bad)} samples whose writer is not in the split's writer list")
        if not samples:
            failures.append(f"[{unit}/{name}] split has no samples")

    # 1. The property that matters: no writer in more than one split.
    for a, b in combinations(SPLIT_NAMES, 2):
        overlap = set(splits[a]) & set(splits[b])
        status = "OK" if not overlap else f"FAIL {sorted(overlap)}"
        print(f"writer overlap {a}/{b}: {len(overlap)}  {status}")
        if overlap:
            failures.append(f"[{unit}] writers in both {a} and {b}: {sorted(overlap)}")
        shared_samples = sample_ids[a] & sample_ids[b]
        if shared_samples:
            failures.append(f"[{unit}] {len(shared_samples)} samples in both {a} and {b}")

    # 2. Coverage: every known writer is in exactly one split.
    union = [w for n in SPLIT_NAMES for w in splits[n]]
    if sorted(union) != corpus.all_writer_ids:
        failures.append(f"[{unit}] union of splits != all writers in forms.txt")

    # 3. The JSON on disk is what the seed produces (auditability / determinism).
    on_disk, meta = read_split_files(splits_dir)
    if on_disk != make_writer_splits(corpus.all_writer_ids, seed, fractions):
        failures.append(f"[{unit}] {splits_dir}/*.json differ from the split re-derived from seed={seed}")
    print(f"split files: {splits_dir}  seed={meta['seed']}  fractions={meta['fractions']}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="IAM root (default: data.iam.root from config)")
    parser.add_argument("--splits-dir", default=None, help="where the writer-ID JSON lives (default: config)")
    parser.add_argument("--unit", choices=[*UNITS, "both"], default="both")
    parser.add_argument("--regenerate", action="store_true", help="rewrite the split JSON files")
    args = parser.parse_args()

    cfg = load_config()["data"]["iam"]
    splits_dir = Path(args.splits_dir) if args.splits_dir else resolve_path(cfg["splits_dir"])
    units = UNITS if args.unit == "both" else (args.unit,)

    failures: list[str] = []
    try:
        for i, unit in enumerate(units):
            # Only the first unit may (re)write the files; the second must agree with them.
            failures += check_unit(unit, args.root, splits_dir, cfg, args.regenerate and i == 0)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:   # split files disagree with the seed/forms.txt, or malformed metadata
        failures.append(str(e))

    print()
    if failures:
        print("SPLIT CHECK FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("SPLIT CHECK PASSED: zero writer overlap, full coverage, files match seed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
