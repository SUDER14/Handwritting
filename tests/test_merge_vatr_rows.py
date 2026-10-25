"""scripts/merge_vatr_rows.py against a FAKE fixture with invented numbers (0.111111 / 0.222222 / 0.333333), built here
from the real "real" rows' writer list. These are NOT VATr results. The merge target is a tmp copy, never results/."""
import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import merge_vatr_rows as M  # noqa: E402
from src.evaluation.protocol_cli import TABLE_COLUMNS  # noqa: E402

CRNN = "crnn_20260922T003942Z_d1f87b49_cpu_r1"      # the pins in config.yaml
WID = "writerid_20260921T180911Z_ed897d02_cpu_r1"
FAKE_CER, FAKE_WID, FAKE_HWD = 0.111111, 0.222222, 0.333333


def build_fake_colab_output(dest: Path, writers: list[str]) -> None:
    """FAKE Colab output: invented numbers, obviously sentinel values, marked FAKE_FIXTURE inside every file."""
    dest.mkdir(parents=True)
    rows = []
    for K in (1, 5):
        rid = f"FAKE00000000Z_deadbeef_protocol_vatr_K{K}"
        agg = {"n_writers": 20, "all_cer": FAKE_CER, "all_wid_dist": FAKE_WID, "all_hwd": FAKE_HWD, "all_n_writers": 20,
               "all_n_lines": 200, "oov_cer": None, "oov_wid_dist": None, "oov_hwd": None, "oov_n_writers": 0,
               "oov_n_lines": 0, "n_failed_generations": 0}
        rec = {"FAKE_FIXTURE": "INVENTED NUMBERS FOR A TEST. NOT A VATr RESULT.", "run_id": rid, "split": "test",
               "backend": "vatr", "K": K, "crnn_run": CRNN, "writer_id": {"run_id": WID, "is_stub": False, "usable": True},
               "aggregate": agg, "writers": [{"writer": w} for w in writers]}
        (dest / rid).mkdir()
        (dest / rid / "metrics.json").write_text(json.dumps(rec, indent=2) + "\n")
        rows.append({"run_id": rid, "utc": "1999-01-01T00:00:00+00:00", "git_sha": "deadbeef", "git_dirty": False,
                     "split": "test", "subset": "FAKE", "cpu_reduced": False, "backend": "vatr", "K": K, "n_writers": 20,
                     "cer": FAKE_CER, "wid_dist": FAKE_WID, "hwd": FAKE_HWD, "oov_n_writers": 0, "oov_cer": "",
                     "oov_wid_dist": "", "oov_hwd": "", "n_failed_generations": 0, "crnn_run": CRNN, "writer_id_run": WID,
                     "metrics_path": "/content/FAKE/metrics.json"})
    with (dest / "protocol_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def setup(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    shutil.copy2(ROOT / "results" / "protocol_table.csv", res / "protocol_table.csv")
    real = sorted(ROOT.glob("results/*_protocol_real_K1"))
    assert real, "needs the committed real-row run dirs"
    for d in ROOT.glob("results/*_protocol_real_K*"):
        shutil.copytree(d, res / d.name)
    writers = [w["writer"] for w in json.loads((real[0] / "metrics.json").read_text())["writers"]]
    colab = tmp_path / "colab"
    build_fake_colab_output(colab, writers)
    return res / "protocol_table.csv", colab


def run(table, colab, **kw):
    return M.merge(colab / "protocol_table.csv", colab, table, **kw)


def rewrite_row(colab, index=0, **changes):
    p = colab / "protocol_table.csv"
    rows = list(csv.DictReader(p.open(newline="")))
    rows[index].update(changes)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_valid_fake_rows_merge_and_paths_are_rewritten(setup):
    table, colab = setup
    n_before = len(list(csv.DictReader(table.open())))
    ids = run(table, colab)
    rows = list(csv.DictReader(table.open()))
    assert len(ids) == 2 and len(rows) == n_before + 2
    assert [r["backend"] for r in rows[-2:]] == ["vatr", "vatr"]
    assert str(table.parent) in rows[-1]["metrics_path"] and (table.parent / ids[-1] / "metrics.json").exists()


def test_dry_run_writes_nothing(setup):
    table, colab = setup
    before = table.read_text()
    run(table, colab, dry_run=True)
    assert table.read_text() == before


def test_unpinned_crnn_is_refused(setup):
    table, colab = setup
    rewrite_row(colab, crnn_run="crnn_20991231T000000Z_deadbeef_cpu_r9")
    with pytest.raises(M.MergeError, match="unpinned"):
        run(table, colab)


def test_unpinned_writer_id_in_metrics_json_is_refused(setup):
    table, colab = setup
    mj = next(colab.glob("*K1/metrics.json"))
    rec = json.loads(mj.read_text())
    rec["writer_id"]["run_id"] = "writerid_other"
    mj.write_text(json.dumps(rec))
    with pytest.raises(M.MergeError, match="pins"):
        run(table, colab)


@pytest.mark.parametrize("col,val,match", [("K", "7", "K=7"), ("n_writers", "19", "n_writers"),
                                           ("backend", "baseline", "only 'vatr'"), ("split", "val", "split"),
                                           ("cer", "0.5", "aggregate")])
def test_bad_values_are_refused(setup, col, val, match):
    table, colab = setup
    rewrite_row(colab, **{col: val})
    with pytest.raises(M.MergeError, match=match):
        run(table, colab)


def test_wrong_columns_refused(setup):
    table, colab = setup
    p = colab / "protocol_table.csv"
    p.write_text(p.read_text().replace("wid_dist", "writer_dist", 1))
    with pytest.raises(M.MergeError, match="column"):
        run(table, colab)


def test_writers_outside_the_seeded_subset_refused(setup):
    table, colab = setup
    mj = next(colab.glob("*K1/metrics.json"))
    rec = json.loads(mj.read_text())
    rec["writers"][0]["writer"] = "999999"
    mj.write_text(json.dumps(rec))
    with pytest.raises(M.MergeError, match="outside the seeded subset"):
        run(table, colab)


def test_duplicate_merge_refused(setup):
    table, colab = setup
    run(table, colab)
    with pytest.raises(M.MergeError, match="already"):
        run(table, colab)


def test_dirty_rows_refused_unless_allowed(setup):
    table, colab = setup
    rewrite_row(colab, git_dirty="True")
    with pytest.raises(M.MergeError, match="dirty"):
        run(table, colab)
    assert run(table, colab, allow_dirty=True)


def test_nothing_is_written_when_one_row_fails(setup):
    table, colab = setup
    rewrite_row(colab, index=1, crnn_run="unpinned")
    before = table.read_text()
    with pytest.raises(M.MergeError):
        run(table, colab)
    assert table.read_text() == before
