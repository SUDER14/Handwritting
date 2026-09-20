"""Tests for the IAM loader and the writer-disjoint split (src/data/iam.py).

Runs against a fake IAM tree (tests/iam_fixture.py): the real, licensed dataset
is not in the repo, so this verifies parsing/grouping/splitting logic, not
anything about real IAM data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.iam import (
    SPLIT_NAMES,
    IAMCorpus,
    form_id_of,
    get_writer_splits,
    image_path_for,
    load_iam_splits,
    make_writer_splits,
    resize_to_height,
)
from src.utils.config import load_config
from tests.iam_fixture import build_fake_iam

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake_iam(tmp_path_factory):
    root = tmp_path_factory.mktemp("iam")
    info = build_fake_iam(root, n_writers=12, forms_per_writer=2, lines_per_form=3)
    return root, info


def _cfg(tmp_path: Path, seed: int = 1337) -> dict:
    cfg = json.loads(json.dumps(load_config()))  # deep copy; load_config() is cached
    cfg["data"]["iam"]["split_seed"] = seed
    return cfg


# -- id / path helpers --------------------------------------------------------

def test_id_helpers():
    assert form_id_of("a01-000u-00", "lines") == "a01-000u"
    assert form_id_of("a01-000u-00-03", "words") == "a01-000u"
    assert image_path_for(Path("r"), "lines", "a01-000u-00") == Path("r/lines/a01/a01-000u/a01-000u-00.png")
    assert image_path_for(Path("r"), "words", "a01-000u-00-03") == Path("r/words/a01/a01-000u/a01-000u-00-03.png")


# -- the split ----------------------------------------------------------------

def test_split_is_disjoint_and_covers_all_writers():
    writers = [f"{i:03d}" for i in range(50)]
    s = make_writer_splits(writers, 7, FRACTIONS)
    assert sorted(s["train"] + s["val"] + s["test"]) == writers
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                assert not set(s[a]) & set(s[b])
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (40, 5, 5)


def test_split_is_deterministic_and_seed_dependent():
    writers = [f"{i:03d}" for i in range(50)]
    a = make_writer_splits(writers, 7, FRACTIONS)
    assert a == make_writer_splits(list(reversed(writers)), 7, FRACTIONS)  # order-independent
    assert a != make_writer_splits(writers, 8, FRACTIONS)


def test_split_rejects_bad_inputs():
    with pytest.raises(ValueError):
        make_writer_splits(["1", "2"], 0, FRACTIONS)                       # < 3 writers
    with pytest.raises(ValueError):
        make_writer_splits([str(i) for i in range(20)], 0, {"train": 0.5, "val": 0.1, "test": 0.1})


def test_split_files_written_verified_and_tamper_detected(tmp_path):
    writers = [f"{i:03d}" for i in range(30)]
    splits_dir = tmp_path / "splits"
    first = get_writer_splits(writers, 5, FRACTIONS, splits_dir)
    payload = json.loads((splits_dir / "val.json").read_text())
    assert payload["writer_ids"] == first["val"] and payload["seed"] == 5 and payload["split"] == "val"
    assert get_writer_splits(writers, 5, FRACTIONS, splits_dir) == first          # re-read + verified
    with pytest.raises(ValueError, match="seed"):
        get_writer_splits(writers, 6, FRACTIONS, splits_dir)                       # config drift
    payload["writer_ids"] = payload["writer_ids"][:-1] + [first["train"][0]]       # hand edit
    (splits_dir / "val.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="do not match"):
        get_writer_splits(writers, 5, FRACTIONS, splits_dir)


# -- corpus + dataset ---------------------------------------------------------

def test_corpus_groups_by_writer_and_skips_bad_rows(fake_iam):
    root, info = fake_iam
    corpus = IAMCorpus(root=root, unit="lines")
    assert corpus.all_writer_ids == sorted(info["writers"])
    assert corpus.skipped == {"status_err": info["n_err"], "unknown_form": 1, "missing_image": info["n_missing"]}
    for writer, ids in info["line_ids_by_writer"].items():
        assert [s.sample_id for s in corpus.samples_by_writer[writer]] == sorted(ids)
        assert all(s.writer_id == writer for s in corpus.samples_by_writer[writer])
    # forms.txt is the only source of writer identity: forms of one writer share it.
    a_form = next(iter(info["form_to_writer"]))
    assert any(s.form_id == a_form for s in corpus.samples_by_writer[info["form_to_writer"][a_form]])


def test_transcription_word_gaps_become_spaces(fake_iam):
    root, _ = fake_iam
    corpus = IAMCorpus(root=root, unit="lines")
    texts = [s.transcription for ss in corpus.samples_by_writer.values() for s in ss]
    assert all("|" not in t for t in texts) and any(" " in t for t in texts)
    words = IAMCorpus(root=root, unit="words")
    assert all(" " not in s.transcription for ss in words.samples_by_writer.values() for s in ss)


def test_datasets_are_writer_disjoint_and_complete(fake_iam, tmp_path):
    root, info = fake_iam
    ds = load_iam_splits(unit="lines", config=_cfg(tmp_path), root=root, splits_dir=tmp_path / "splits")
    writer_sets = {n: {w for _, _, w in ds[n]} for n in SPLIT_NAMES}
    for a in SPLIT_NAMES:
        for b in SPLIT_NAMES:
            if a < b:
                assert not writer_sets[a] & writer_sets[b]
        assert writer_sets[a] <= set(ds[a].writer_ids)
    total = sum(len(ds[n]) for n in SPLIT_NAMES)
    assert total == sum(len(v) for v in info["line_ids_by_writer"].values())
    assert all(len(ds[n]) > 0 for n in SPLIT_NAMES)


def test_split_stable_when_only_some_images_are_present(fake_iam, tmp_path):
    """Writers come from forms.txt, so removing images must not move anyone between splits."""
    root, _ = fake_iam
    full = load_iam_splits(unit="lines", config=_cfg(tmp_path), root=root, splits_dir=tmp_path / "a")
    corpus = IAMCorpus(root=root, unit="lines")
    assert {n: full[n].writer_ids for n in SPLIT_NAMES} == make_writer_splits(
        corpus.all_writer_ids, 1337, FRACTIONS)


def test_images_have_fixed_height_variable_width_and_preserved_aspect(fake_iam, tmp_path):
    root, _ = fake_iam
    cfg = _cfg(tmp_path)
    ds = load_iam_splits(unit="lines", config=cfg, root=root, splits_dir=tmp_path / "s")["train"]
    height = cfg["data"]["iam"]["image_height"]
    widths = set()
    for i in range(len(ds)):
        img, text, writer = ds[i]
        native = ds.read_native(i)
        assert img.dtype == np.uint8 and img.ndim == 2 and img.shape[0] == height
        assert abs(img.shape[1] / height - native.shape[1] / native.shape[0]) < 1.0 / height + 1e-9
        assert isinstance(text, str) and isinstance(writer, str)
        widths.add(img.shape[1])
    assert len(widths) > 1   # width really is variable


def test_resize_to_height_upscale_and_downscale():
    for h, w, target in [(100, 300, 50), (20, 60, 64), (64, 1, 32)]:
        out = resize_to_height(np.zeros((h, w), np.uint8), target)
        assert out.shape[0] == target and out.shape[1] >= 1


def test_missing_iam_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="IAM metadata not found"):
        IAMCorpus(root=tmp_path / "nope")


# -- the CLI audit ------------------------------------------------------------

def test_check_split_script_passes_and_reports_counts(fake_iam, tmp_path):
    root, _ = fake_iam
    r = subprocess.run(
        [sys.executable, "scripts/check_split.py", "--root", str(root), "--splits-dir", str(tmp_path / "s")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SPLIT CHECK PASSED" in r.stdout and "writer overlap train/val: 0" in r.stdout


def test_check_split_script_fails_on_overlap(fake_iam, tmp_path):
    root, _ = fake_iam
    splits = tmp_path / "s"
    args = [sys.executable, "scripts/check_split.py", "--root", str(root), "--splits-dir", str(splits), "--unit", "lines"]
    assert subprocess.run(args, cwd=REPO_ROOT, capture_output=True).returncode == 0
    val = json.loads((splits / "val.json").read_text())
    train = json.loads((splits / "train.json").read_text())
    val["writer_ids"].append(train["writer_ids"][0])          # leak one train writer into val
    (splits / "val.json").write_text(json.dumps(val))
    r = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "SPLIT CHECK FAILED" in r.stdout and "do not match" in r.stdout


def test_check_split_script_reports_missing_data(tmp_path):
    r = subprocess.run([sys.executable, "scripts/check_split.py", "--root", str(tmp_path / "none")],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "IAM metadata not found" in r.stderr
