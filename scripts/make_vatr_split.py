"""Write the canonical writer split (data/splits/vatr/{train,val,test}.json) from the VATr/HWT IAM ground-truth files.

Source: https://github.com/aimagelab/VATr  Groundtruth/gan.iam.tr_va.gt.filter27 (train+val) and
gan.iam.test.gt.filter27 (test). Each row is ``<writer_id>,<word_id> <text>``; writer ids are the same as forms.txt's.
VATr has no separate val list, so val = the writers of Teklia/IAM-line's validation split (a subset of VATr's train+val
pool; checked), train = the rest of train+val, test = VATr's test. Requires data/processed/teklia_join/retained.csv.

Usage:  python scripts/make_vatr_split.py [--out data/splits/vatr]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import SPLIT_ALGORITHM, SPLIT_NAMES, parse_forms  # noqa: F401
from src.utils.config import resolve_path

BASE = "https://raw.githubusercontent.com/aimagelab/VATr/main/Groundtruth/gan.iam.{}.gt.filter27"


def writers_of(name: str) -> set[str]:
    text = urllib.request.urlopen(BASE.format(name), timeout=60).read().decode("utf-8")
    return {ln.split(",", 1)[0] for ln in text.splitlines() if ln.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/splits/vatr")
    ap.add_argument("--forms", default="forms.txt")
    ap.add_argument("--join", default="data/processed/teklia_join/retained.csv")
    args = ap.parse_args()

    known = set(parse_forms(Path(args.forms)).values())
    tr_va, test = writers_of("tr_va"), writers_of("test")
    with open(args.join, encoding="utf-8") as f:
        val = {r["writer_id"] for r in csv.DictReader(f) if r["teklia_split"] == "validation"}
    assert not (tr_va & test), "VATr train/val and test writers overlap"
    assert val <= tr_va, f"Teklia val writers outside VATr train+val: {sorted(val - tr_va)}"
    assert (tr_va | test) <= known, "VATr writers unknown to forms.txt"
    splits = {"train": sorted(tr_va - val), "val": sorted(val), "test": sorted(test)}

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        payload = {
            "split": name,
            "source": "VATr/HWT Groundtruth gan.iam.{tr_va,test}.gt.filter27 (aimagelab/VATr); "
                      "val = Teklia/IAM-line validation writers",
            "num_writers": len(splits[name]),
            "writer_ids": splits[name],
        }
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(splits[name])} writers")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
