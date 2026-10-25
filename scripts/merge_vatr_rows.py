"""Append the Colab VATr protocol rows (Task 3.5) to results/protocol_table.csv -- after validating them.

    python scripts/merge_vatr_rows.py --colab-table <colab>/protocol_table.csv --colab-results <colab results dir> [--dry-run]

Everything is validated BEFORE anything is written (all or nothing); a row from unpinned instruments is refused:
  * the CSV header equals TABLE_COLUMNS exactly; only backend == vatr rows are merged;
  * K is allowed (config evaluation.protocol.ks; --allow-sweep adds the Task 5.1 values 3/10/15), split == test, and
    n_writers equals the configured subset size (sweep K's may score fewer writers: 1..n_writers, as recorded in the run);
  * crnn_run and writer_id_run equal the PINS in config (evaluation.crnn_run / writer_id_run), AND the run's metrics.json
    says the same, with a non-stub, usable writer-ID instrument;
  * the metrics.json of every row exists and its aggregate equals the CSV numbers; its scored writers are a subset of the
    seeded 20-writer subset of the existing "real" rows; no NaN/inf;
  * the code that produced the row was clean (git_dirty False) unless --allow-dirty;
  * no duplicate run_id or (backend, K) already in the table.
Merged runs' directories are copied into results/ and metrics_path is rewritten to the local path.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.protocol_cli import TABLE_COLUMNS  # noqa: E402
from src.utils.config import load_config  # noqa: E402

SWEEP_KS = {3, 10, 15}


class MergeError(Exception):
    pass


def _read_table(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def reference_writers(table_rows: list[dict], results_dir: Path) -> set[str]:
    """The seeded 20-writer subset, taken from the existing REAL rows' metrics.json (scored writers)."""
    for row in table_rows:
        if row["backend"] == "real":
            mp = results_dir / row["run_id"] / "metrics.json"
            if mp.exists():
                return {w["writer"] for w in json.loads(mp.read_text(encoding="utf-8"))["writers"]}
    raise MergeError("no existing 'real' row with a readable metrics.json to take the writer subset from")


def validate(colab_table: Path, colab_results: Path, table: Path, cfg: dict, *, allow_sweep: bool,
             allow_dirty: bool) -> list[tuple[dict, Path]]:
    ev = cfg["evaluation"]
    pc = ev["protocol"]
    pins = {"crnn_run": Path(ev["crnn_run"]).name, "writer_id_run": Path(ev["writer_id_run"]).name}
    allowed_ks = set(pc["ks"]) | (SWEEP_KS if allow_sweep else set())
    header, rows = _read_table(colab_table)
    if header != TABLE_COLUMNS:
        missing, extra = sorted(set(TABLE_COLUMNS) - set(header)), sorted(set(header) - set(TABLE_COLUMNS))
        raise MergeError(f"column names differ from TABLE_COLUMNS: missing {missing}, unexpected {extra}"
                         if missing or extra else f"column order differs: {header}")
    _, existing = _read_table(table)
    ref_writers = reference_writers(existing, table.parent)
    if not rows:
        raise MergeError("the Colab table has no rows")
    seen_ids = {r["run_id"] for r in existing}
    seen_bk = {(r["backend"], r["K"]) for r in existing}
    out: list[tuple[dict, Path]] = []
    for row in rows:
        rid = row["run_id"]
        where = f"row {rid}"
        if row["backend"] != "vatr":
            raise MergeError(f"{where}: backend is {row['backend']!r}, only 'vatr' rows are merged")
        K = int(row["K"])
        if K not in allowed_ks:
            raise MergeError(f"{where}: K={K} not in {sorted(allowed_ks)}")
        if row["split"] != "test":
            raise MergeError(f"{where}: split {row['split']!r} != 'test'")
        n_w = int(row["n_writers"])
        if (K in pc["ks"] and n_w != pc["n_writers"]) or not (1 <= n_w <= pc["n_writers"]):
            raise MergeError(f"{where}: n_writers {n_w} (expected {pc['n_writers']}, or fewer for a sweep K)")
        for col, pin in pins.items():
            if row[col] != pin:
                raise MergeError(f"{where}: {col} {row[col]!r} is not the pin {pin!r} -- unpinned instrument, refused")
        if str(row["git_dirty"]).lower() == "true" and not allow_dirty:
            raise MergeError(f"{where}: produced from a dirty tree (git_dirty True); pass --allow-dirty to override")
        if rid in seen_ids or ("vatr", str(K)) in seen_bk:
            raise MergeError(f"{where}: run_id or (vatr, K={K}) already in {table.name}")
        seen_ids.add(rid)
        seen_bk.add(("vatr", str(K)))
        mj = colab_results / rid / "metrics.json"
        if not mj.exists():
            raise MergeError(f"{where}: {mj} not found")
        rec = json.loads(mj.read_text(encoding="utf-8"))
        wid = rec.get("writer_id", {})
        if rec.get("crnn_run") != pins["crnn_run"] or wid.get("run_id") != pins["writer_id_run"]:
            raise MergeError(f"{where}: metrics.json instruments ({rec.get('crnn_run')}, {wid.get('run_id')}) != pins")
        if wid.get("is_stub") or not wid.get("usable"):
            raise MergeError(f"{where}: writer-ID instrument is a stub or not usable")
        if rec.get("backend") != "vatr" or int(rec.get("K", -1)) != K:
            raise MergeError(f"{where}: metrics.json backend/K disagree with the CSV row")
        agg = rec["aggregate"]
        for col, key in (("cer", "all_cer"), ("wid_dist", "all_wid_dist"), ("hwd", "all_hwd")):
            v = float(row[col])
            if not math.isfinite(v) or abs(v - float(agg[key])) > 1e-9:
                raise MergeError(f"{where}: {col} {row[col]} != metrics.json aggregate {agg[key]} (or not finite)")
        if int(agg["n_writers"]) != n_w:
            raise MergeError(f"{where}: n_writers disagrees with metrics.json")
        scored = {w["writer"] for w in rec["writers"]}
        if not scored <= ref_writers:
            raise MergeError(f"{where}: scored writers outside the seeded subset: {sorted(scored - ref_writers)}")
        out.append((row, colab_results / rid))
    return out


def merge(colab_table, colab_results, table, *, allow_sweep=False, allow_dirty=False, dry_run=False,
          cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    ok = validate(Path(colab_table), Path(colab_results), Path(table), cfg, allow_sweep=allow_sweep,
                  allow_dirty=allow_dirty)
    if dry_run:
        return [r["run_id"] for r, _ in ok]
    results_dir = Path(table).parent
    with Path(table).open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        for row, mdir in ok:
            dest = results_dir / row["run_id"]
            shutil.copytree(mdir, dest)
            w.writerow(dict(row, metrics_path=str(dest / "metrics.json")))
    return [r["run_id"] for r, _ in ok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--colab-table", required=True)
    ap.add_argument("--colab-results", required=True)
    ap.add_argument("--table", default=str(ROOT / "results" / "protocol_table.csv"))
    ap.add_argument("--allow-sweep", action="store_true", help="also accept K in {3, 10, 15} (Task 5.1 sweep rows)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    try:
        ids = merge(a.colab_table, a.colab_results, a.table, allow_sweep=a.allow_sweep, allow_dirty=a.allow_dirty,
                    dry_run=a.dry_run)
    except MergeError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(("validated (dry run, nothing written): " if a.dry_run else "merged: ") + ", ".join(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
