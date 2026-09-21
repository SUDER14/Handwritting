"""Write the joined Teklia/IAM-line lines to disk in the layout src/data/iam.py reads (data/iam/, gitignored).

Inputs : data/processed/teklia_join/retained.csv (from scripts/join_teklia.py), forms.txt, HF Teklia/IAM-line.
Outputs: data/iam/ascii/forms.txt          (copy of forms.txt)
         data/iam/ascii/lines.txt          (retained lines only; IAM columns we do not have are zero-filled,
                                            transcription = Teklia text, '|' between words)
         data/iam/lines/<a01>/<a01-000u>/<line_id>.png   (grayscale, dark ink on light paper, native polarity)

Only the ~90% of IAM lines the join could attribute to a writer are materialised; that subset IS the corpus every
later stage uses. Usage:  python scripts/materialize_iam.py [--root data/iam]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import image_path_for
from src.utils.config import load_config, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--join", default="data/processed/teklia_join/retained.csv")
    ap.add_argument("--forms", default="forms.txt")
    args = ap.parse_args()
    root = Path(args.root) if args.root else resolve_path(load_config()["data"]["iam"]["root"])

    from datasets import load_dataset
    teklia = load_dataset("Teklia/IAM-line")
    with open(args.join, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    (root / "ascii").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.forms, root / "ascii" / "forms.txt")

    n = 0
    lines_out = []
    for r in sorted(rows, key=lambda r: r["line_id"]):
        ex = teklia[r["teklia_split"]][int(r["teklia_index"])]
        path = image_path_for(root, "lines", r["line_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            ex["image"].convert("L").save(path)
        text = ex["text"].strip()
        lines_out.append(f"{r['line_id']} ok 0 0 0 0 0 0 {text.replace(' ', '|')}")
        n += 1
    (root / "ascii" / "lines.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"materialised {n} lines under {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
