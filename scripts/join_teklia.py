"""Task 6a + 6: validate forms.txt, then join Teklia/IAM-line to writer IDs via xReniar/IAM-Dataset.

Teklia/IAM-line has only (image, text): no line id, no writer. xReniar/IAM-Dataset has (line_id, line_text) per
form, and forms.txt maps form -> writer. So the join is  Teklia text -> line_id (xReniar) -> form -> writer.

Text is the only key Teklia offers, so a Teklia row is retained only when its match is unambiguous:
  * whitespace-insensitive text key (Teklia tokenises punctuation, xReniar does not); case is significant;
  * candidates are restricted to the writers Teklia's split allows: Teklia's own splits are writer-disjoint and equal
    the canonical VATr/HWT partition (Teklia test = canonical test writers; Teklia train+validation = canonical
    train+val writers), so a Teklia "test" row can only come from a test writer, etc. This uses only the published
    split lists, not the join output;
  * a row is retained when, within its group, exactly one xReniar line_id AND exactly one Teklia row have that key.
    Anything else (unmatched, or the same sentence still shared by several writers of one group) is EXCLUDED and
    logged -- never guessed.

Guards: forms.txt duplicates (same form, different writer => stop); duplicate line_ids in the line source (kept only
if the copies' text is identical, else excluded); final hard assertion that retained line_ids are unique and each has
exactly one writer.

Usage:  python scripts/join_teklia.py [--forms forms.txt] [--out data/processed/teklia_join]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.iam import SPLIT_NAMES, resolve_writer_splits
from src.utils.config import load_config, resolve_path

OFFICIAL = {"writers": 657, "forms": 1539}


def key(text: str) -> str:
    return "".join(text.split())


def validate_forms(path: Path) -> dict[str, str]:
    """Task 6a. Returns {form_id: writer_id}; raises SystemExit on the one blocking condition."""
    seen: dict[str, list[str]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 2:
            raise SystemExit(f"forms.txt: malformed row {raw!r}")
        seen[parts[0]].append(parts[1])
    dups = {f: w for f, w in seen.items() if len(w) > 1}
    conflicts = {f: w for f, w in dups.items() if len(set(w)) > 1}
    print(f"forms.txt: {len(seen)} unique forms, {len(dups)} duplicated form_ids "
          f"({len(conflicts)} with conflicting writers)")
    if conflicts:
        raise SystemExit(f"STOP: same form_id, different writer_ids: {conflicts}")
    forms = {f: w[0] for f, w in seen.items()}
    writers = set(forms.values())
    for what, got in (("forms", len(forms)), ("writers", len(writers))):
        print(f"forms.txt: {what} = {got} (published {OFFICIAL[what]}) {'OK' if got == OFFICIAL[what] else 'MISMATCH'}")
    return forms


def load_line_source(forms: dict[str, str]):
    """xReniar lines -> (kept rows, excluded log). Applies the duplicate-line_id rule."""
    from datasets import load_dataset
    ds = load_dataset("xReniar/IAM-Dataset")["train"]
    by_id: dict[str, list[tuple[str, str]]] = defaultdict(list)   # line_id -> [(form_id, text)]
    for r in ds:
        form_id = r["filename"].rsplit(".", 1)[0]
        for lid, txt in zip(r["line_id"], r["line_text"]):
            by_id[lid].append((form_id, txt))
    rows, excluded = [], []
    for lid, copies in by_id.items():
        if len(copies) > 1:
            same = len({c for c in copies}) == 1
            print(f"DUPLICATE line_id {lid}: {len(copies)} copies, "
                  f"{'identical -> keep one' if same else 'DIFFERENT -> exclude entirely'}: {copies}")
            if not same:
                excluded.append({"ref": lid, "reason": "duplicate_line_id_conflicting_text"})
                continue
        form_id, txt = copies[0]
        if form_id not in forms:
            excluded.append({"ref": lid, "reason": "form_not_in_forms_txt"})
            continue
        rows.append({"line_id": lid, "form_id": form_id, "writer_id": forms[form_id], "text": txt})
    n_dup = sum(1 for c in by_id.values() if len(c) > 1)
    print(f"line source: {sum(len(c) for c in by_id.values())} rows, {len(by_id)} distinct line_ids, "
          f"{n_dup} duplicated line_ids")
    return rows, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forms", default="forms.txt")
    ap.add_argument("--out", default="data/processed/teklia_join")
    args = ap.parse_args()

    forms = validate_forms(Path(args.forms))                      # Task 6a

    src_rows, src_excluded = load_line_source(forms)
    src_by_key: dict[str, list[dict]] = defaultdict(list)
    for r in src_rows:
        src_by_key[key(r["text"])].append(r)

    from datasets import load_dataset
    teklia = load_dataset("Teklia/IAM-line")
    tk_rows = [(s, i, t) for s in teklia for i, t in enumerate(teklia[s]["text"])]   # image column not decoded

    cfg = load_config()["data"]["iam"]
    splits = resolve_writer_splits(cfg, sorted(set(forms.values())))     # canonical (split_source) split
    group_writers = {"test": set(splits["test"]), "trainval": set(splits["train"]) | set(splits["val"])}
    group_of = {"train": "trainval", "validation": "trainval", "test": "test"}     # Teklia split -> writer group
    tk_count = Counter((group_of[s], key(t)) for s, _, t in tk_rows)

    retained, excluded = [], []
    for s, i, t in tk_rows:
        k, g = key(t), group_of[s]
        all_cands = src_by_key.get(k, [])
        cands = [c for c in all_cands if c["writer_id"] in group_writers[g]]
        n_tk = tk_count[(g, k)]
        ref = f"{s}[{i}]"
        if not all_cands:
            excluded.append({"ref": ref, "reason": "unmatched_text", "text": t})
        elif not cands:
            excluded.append({"ref": ref, "reason": "matched_only_outside_teklia_split_writers", "text": t})
        elif len(cands) > 1 or n_tk > 1:
            excluded.append({"ref": ref, "reason": f"ambiguous_text({len(cands)} lines, {n_tk} teklia rows)",
                             "text": t})
        else:
            retained.append({**cands[0], "teklia_split": s, "teklia_index": i})

    # Hard assertions (guard 3).
    ids = [r["line_id"] for r in retained]
    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    assert not dup_ids, f"retained line_ids not unique: {dup_ids[:10]}"
    per_id_writers: dict[str, set[str]] = defaultdict(set)
    for r in retained:
        per_id_writers[r["line_id"]].add(r["writer_id"])
        assert forms[r["form_id"]] == r["writer_id"]
    bad = [i for i, w in per_id_writers.items() if len(w) != 1]
    assert not bad, f"lines without exactly one writer_id: {bad[:10]}"

    w2split = {w: n for n in SPLIT_NAMES for w in splits[n]}
    for r in retained:
        r["split"] = w2split.get(r["writer_id"], "none")                # writers outside the canonical lists

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("retained.csv", retained), ("excluded.csv", excluded + src_excluded)):
        fields = sorted({f for d in data for f in d}) if data else ["ref"]
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    n = len(tk_rows)
    print(f"\nTeklia rows: {n}   retained: {len(retained)} ({len(retained) / n:.1%})   excluded: {len(excluded)}")
    for reason, c in Counter(e["reason"].split("(")[0] for e in excluded).items():
        print(f"  excluded {reason}: {c}")
    print(f"line-source rows excluded: {len(src_excluded)}")
    print(f"{'split':<7}{'writers (split)':>16}{'writers (retained)':>20}{'lines':>8}")
    for name in SPLIT_NAMES:
        rw = {r["writer_id"] for r in retained if r["split"] == name}
        print(f"{name:<7}{len(splits[name]):>16}{len(rw):>20}{sum(r['split'] == name for r in retained):>8}")
    print(f"retained lines whose writer is outside the canonical split: {sum(r['split'] == 'none' for r in retained)}")
    print("ASSERTIONS PASSED: retained line_ids unique, exactly one writer_id each.")
    print(f"wrote {out}/retained.csv, excluded.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
