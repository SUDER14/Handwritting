"""Writer-ID model, losses, augmentation, retrieval metrics, run directories, embedder, calibration.

All data here is a fake IAM tree of Hershey-font renders (tests/iam_fixture.py): these tests verify MECHANICS
(shapes, invariances, exact metric arithmetic, never-overwrite), not anything about real handwriting.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.iam import load_iam_splits
from src.evaluation.writer_id import TrainedWriterEmbedder, get_writer_embedder
from src.features.style_extractor import StyleFeatureExtractor
from src.utils.config import load_config, resolve_path
from src.writer_id.data import (
    LineStore, augment_window, canonical_from_binary, canonical_from_iam_gray, random_window,
)
from src.writer_id.model import WriterEncoder, WriterIDNet, nt_xent_multi_positive
from src.writer_id.retrieval import embed_line, retrieval_metrics
from src.writer_id.runs import best_checkpoint, latest_run, new_run_dir, write_once
from tests.iam_fixture import build_fake_iam

REPO = Path(__file__).resolve().parents[1]
CFG = load_config()
AUG = CFG["writer_id"]["augment"]


# -- model ---------------------------------------------------------------------------------------

def test_encoder_gives_256d_embedding_for_any_width_and_head_matches_train_writers():
    enc = WriterEncoder(**{"widths": [32, 64, 128, 256], "blocks": [2, 2, 2, 2], "stem_stride": 2})
    enc.eval()
    for w in (64, 200, 512):
        assert enc(torch.zeros(3, 1, 64, w)).shape == (3, 256)
    net = WriterIDNet(37)
    emb, logits = net(torch.zeros(2, 1, 64, 160))
    assert emb.shape == (2, 256) and logits.shape == (2, 37)
    assert not any("pretrained" in n for n, _ in net.named_modules())      # from scratch by construction


def test_encoder_is_trainable_end_to_end():
    net = WriterIDNet(4)
    x = torch.rand(8, 1, 64, 96)
    emb, logits = net(x)
    (logits.logsumexp(1).mean() + emb.pow(2).mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters() if p.requires_grad)


# -- NT-Xent --------------------------------------------------------------------------------------

def test_ntxent_matches_the_textbook_formula_with_two_views():
    torch.manual_seed(0)
    z = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])            # views (0,3), (1,4), (2,5)
    t = 0.2
    zn = torch.nn.functional.normalize(z, dim=1)
    sim = zn @ zn.T / t
    expected = []
    for i, j in [(0, 3), (3, 0), (1, 4), (4, 1), (2, 5), (5, 2)]:
        denom = sum(torch.exp(sim[i, k]) for k in range(6) if k != i)
        expected.append(-torch.log(torch.exp(sim[i, j]) / denom))
    assert nt_xent_multi_positive(z, labels, t).item() == pytest.approx(torch.stack(expected).mean().item(), rel=1e-5)


def test_ntxent_prefers_tight_same_writer_clusters_and_skips_positive_free_anchors():
    labels = torch.tensor([0, 0, 1, 1])
    tight = torch.tensor([[1.0, 0], [1, 0.01], [0, 1], [0.01, 1]])
    mixed = torch.tensor([[1.0, 0], [0, 1], [1, 0.01], [0.01, 1]])
    assert nt_xent_multi_positive(tight, labels, 0.1) < nt_xent_multi_positive(mixed, labels, 0.1)
    z = torch.randn(3, 4, requires_grad=True)
    assert nt_xent_multi_positive(z, torch.tensor([0, 1, 2]), 0.1).item() == 0.0     # nobody has a positive


# -- augmentation must not touch the style signal ------------------------------------------------------

def _slanted_text_window(slant_px: float) -> np.ndarray:
    """A 64 x 224 window of thick vertical-ish strokes leaning by `slant_px` across their height."""
    img = np.zeros((64, 224), np.uint8)
    for x in range(20, 210, 26):
        cv2.line(img, (x, 56), (int(x + slant_px), 8), 255, 5)
    return img


def _stats(win: np.ndarray) -> tuple[float, float]:
    """(slant in deg, stroke width in px). Slant is measured PER STROKE (connected component) with the extractor's
    own estimator and averaged: over the whole window its PCA axis is the ~190 px horizontal spread of the eight
    strokes, so the result sat at +/-90 deg and flipped sign under any perturbation (the base window read 0.0 only
    through the estimator's dy == 0 special case)."""
    ex = StyleFeatureExtractor()
    n, lab = cv2.connectedComponents((win > 127).astype(np.uint8))
    slants = [ex._slant_deg(((lab == k) * 255).astype(np.uint8)) for k in range(1, n) if (lab == k).sum() > 20]
    return float(np.mean(slants)), ex._stroke_width(win)


def test_augmentation_adds_no_slant_and_no_stroke_width_change():
    """Mean over many augmentations: slant and stroke width stay where they were (a shear/dilate/erode augmentation
    would move them systematically). Rotation is +/-1 deg and zero-mean, so the mean slant barely moves."""
    rng = np.random.default_rng(0)
    for slant_px in (0.0, 14.0):
        base = _slanted_text_window(slant_px)
        s0, w0 = _stats(base)
        outs = [_stats(augment_window(base, AUG, rng)) for _ in range(60)]
        assert abs(np.mean([o[0] for o in outs]) - s0) < 1.5, (slant_px, s0)
        assert 0.9 < np.mean([o[1] for o in outs]) / w0 < 1.1, (slant_px, w0)
    assert "shear" not in json.dumps(AUG).lower() and "dilate" not in json.dumps(AUG).lower()


def test_augmentation_actually_changes_the_image_and_is_seeded():
    base = _slanted_text_window(6.0)
    a = augment_window(base, AUG, np.random.default_rng(1))
    b = augment_window(base, AUG, np.random.default_rng(1))
    c = augment_window(base, AUG, np.random.default_rng(2))
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, base)


def test_random_window_shapes_and_padding():
    rng = np.random.default_rng(0)
    long_line = np.zeros((64, 500), np.uint8); long_line[20:40, 100:400] = 255
    assert random_window(long_line, 200, rng).shape == (64, 200)
    short = np.full((64, 90), 255, np.uint8)
    win = random_window(short, 200, rng)
    assert win.shape == (64, 200) and win[:, :90].min() == 255 and win[:, 90:].max() == 0


def test_generated_and_real_crops_reach_the_same_canonical_form():
    """A scan (dark ink on paper) and a binary render of the same thing canonicalise identically."""
    gray = np.full((120, 400), 235, np.uint8)
    cv2.putText(gray, "abc", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, 20, 6)
    binary = ((gray < 128) * 255).astype(np.uint8)
    a, b = canonical_from_iam_gray(gray, 64), canonical_from_binary(binary, 64)
    assert a.shape == b.shape and a.shape[0] == 64
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 2


# -- retrieval metrics: exact arithmetic --------------------------------------------------------------------

def _brute_force(emb, writers, forms, exclude_same_form):
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    top1, aps = [], []
    for i in range(len(e)):
        cand = [j for j in range(len(e)) if j != i and not (exclude_same_form and forms[j] == forms[i])]
        pos = [j for j in cand if writers[j] == writers[i]]
        if not pos:
            continue
        order = sorted(cand, key=lambda j: -float(e[i] @ e[j]))
        hits = [writers[j] == writers[i] for j in order]
        top1.append(hits[0])
        prec = [(k + 1) / (r + 1) for k, r in enumerate([r for r, h in enumerate(hits) if h])]
        aps.append(np.mean(prec))
    return np.mean(top1), np.mean(aps), len(top1)


@pytest.mark.parametrize("exclude", [True, False])
def test_retrieval_metrics_match_brute_force(exclude):
    rng = np.random.default_rng(3)
    writers = [f"w{i // 6}" for i in range(48)]                    # 8 writers x 6 lines
    forms = [f"{w}-f{(i % 6) // 2}" for i, w in enumerate(writers)]   # 3 forms per writer
    emb = rng.normal(size=(48, 16)) + np.repeat(rng.normal(size=(8, 16)), 6, axis=0) * 1.5
    m = retrieval_metrics(emb, writers, forms, exclude_same_form=exclude)
    t1, mp, nq = _brute_force(emb, writers, forms, exclude)
    assert m["top1"] == pytest.approx(t1) and m["map"] == pytest.approx(mp) and m["n_queries"] == nq
    assert 0 < m["chance_top1"] < 0.2


def test_retrieval_perfect_and_chance_and_same_form_effect():
    writers = [w for w in "abcd" for _ in range(4)]
    forms = [f"{w}{k // 2}" for w in "abcd" for k in range(4)]
    onehot = np.repeat(np.eye(4), 4, axis=0)
    perfect = retrieval_metrics(onehot + 1e-3, writers, forms)
    assert perfect["top1"] == 1.0 and perfect["map"] == 1.0
    rng = np.random.default_rng(0)
    random = [retrieval_metrics(rng.normal(size=(16, 8)), writers, forms)["top1"] for _ in range(40)]
    assert 0.0 <= np.mean(random) < 0.4                             # near chance (1/5 here), nowhere near 1
    # a "page fingerprint" embedding: same form -> same vector. Cross-form protocol sees through it; same-form does not.
    page = np.repeat(np.eye(8), 2, axis=0)
    assert retrieval_metrics(page, writers, forms, exclude_same_form=False)["top1"] == 1.0
    assert retrieval_metrics(page, writers, forms, exclude_same_form=True)["top1"] < 0.5


def test_retrieval_needs_a_cross_form_positive():
    with pytest.raises(ValueError, match="no query"):
        retrieval_metrics(np.random.rand(4, 3), ["a", "a", "b", "b"], ["f1", "f1", "f2", "f2"])


# -- run directories --------------------------------------------------------------------------------------------

def test_run_dirs_are_never_overwritten(tmp_path):
    run_id, path = new_run_dir("abc12345", False, "t", tmp_path)
    assert path.parent == tmp_path and run_id.startswith("writerid_") and run_id.endswith("_abc12345_t")
    write_once(path / "x.json", "1")
    with pytest.raises(FileExistsError):
        write_once(path / "x.json", "2")
    with pytest.raises(FileExistsError):
        # same stamp/sha/name in the same second must not silently reuse the directory
        import src.writer_id.runs as runs
        real = runs.datetime
        class Frozen(real):  # noqa: E306
            @classmethod
            def now(cls, tz=None):
                return real.strptime(run_id.split("_")[1], "%Y%m%dT%H%M%SZ").replace(tzinfo=tz)
        runs.datetime = Frozen
        try:
            new_run_dir("abc12345", False, "t", tmp_path)
        finally:
            runs.datetime = real


# -- end to end on a fake IAM tree: train -> checkpoints -> retrieval -> embedder -> evaluate hook --------------------

@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wid")
    build_fake_iam(tmp / "iam", n_writers=20, forms_per_writer=2, lines_per_form=3, unique_styles=True)
    ckpt_root = tmp / "ckpt"
    r = subprocess.run(
        [sys.executable, "scripts/train_writer_id.py", "--iam-root", str(tmp / "iam"), "--splits-dir", str(tmp / "splits"),
         "--checkpoint-root", str(ckpt_root), "--epochs", "2", "--steps-per-epoch", "6", "--run-name", "unit"],
        cwd=REPO, capture_output=True, text=True, timeout=900)
    return tmp, ckpt_root, r


def test_training_script_runs_writes_the_run_record_and_reports_test_retrieval(trained_run):
    tmp, root, r = trained_run
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = [p for p in root.iterdir() if p.is_dir()]
    assert len(runs) == 1
    run = runs[0]
    names = {p.name for p in run.iterdir()}
    assert {"config.json", "train_log.csv", "last.pt", "best.pt", "run.json", "test_metrics.json"} <= names
    assert best_checkpoint(run) is not None and latest_run(root) == run
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["kind"] == "writer_id" and cfg["writer_id"]["loss"]["ntxent_weight"] == CFG["writer_id"]["loss"]["ntxent_weight"]
    tm = json.loads((run / "test_metrics.json").read_text())
    assert tm["split"] == "test" and 0 <= tm["top1"] <= 1 and tm["n_queries"] > 0 and "chance_top1" in tm
    # the loud verdict is printed either way: a usable line or the NOT USABLE banner
    assert "TEST writers (unseen)" in r.stdout
    if tm["top1"] < CFG["writer_id"]["eval"]["min_usable_top1"]:
        assert "NOT USABLE" in r.stdout


def test_a_second_training_run_gets_its_own_directory(trained_run):
    tmp, root, _ = trained_run
    r = subprocess.run(
        [sys.executable, "scripts/train_writer_id.py", "--iam-root", str(tmp / "iam"), "--splits-dir", str(tmp / "splits"),
         "--checkpoint-root", str(root), "--epochs", "1", "--steps-per-epoch", "2", "--run-name", "second", "--skip-test"],
        cwd=REPO, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert len(dirs) == 2 and dirs[0].endswith("_unit") and dirs[1].endswith("_second")
    assert not (root / dirs[1] / "test_metrics.json").exists()


def test_trained_embedder_is_a_real_writer_embedder_behind_the_same_interface(trained_run):
    _, root, _ = trained_run
    run = next(p for p in root.iterdir() if p.name.endswith("_unit"))
    emb = TrainedWriterEmbedder(run, CFG)
    d = emb.describe()
    assert d["is_stub"] is False and d["run_id"] == run.name and d["usable"] in (True, False)
    assert d["min_usable_top1"] == 0.6 and d["test_top1"] is not None
    img = np.zeros((90, 700), np.uint8)
    cv2.putText(img, "hello there", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, 255, 3)
    v = emb.embed(img)
    assert v.shape == (256,) and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    assert emb.distance(img, img) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        emb.embed(np.zeros((50, 50), np.uint8))
    assert get_writer_embedder("trained", CFG, run_dir=run).name == d["name"]


def test_usable_flag_follows_the_threshold(trained_run, tmp_path):
    _, root, _ = trained_run
    run = next(p for p in root.iterdir() if p.name.endswith("_unit"))
    cfg = copy.deepcopy(CFG)
    emb = TrainedWriterEmbedder(run, cfg)
    for top1, expect in ((0.59, False), (0.6, True), (0.95, True)):
        emb.test_metrics = {**emb.test_metrics, "top1": top1}
        assert emb.usable is expect
    emb.test_metrics = None
    assert emb.usable is False                       # no recorded test metrics -> never "usable"


def test_trained_embedder_never_falls_back_to_the_stub(tmp_path, monkeypatch):
    import src.writer_id.runs as runs
    monkeypatch.setattr(runs, "CHECKPOINT_ROOT", str(tmp_path / "none"))
    cfg = copy.deepcopy(CFG)
    cfg["evaluation"]["writer_id_run"] = None       # no pin and no run on disk (config.yaml now pins the GATE-1 run)
    with pytest.raises(FileNotFoundError, match="scripts/train_writer_id.py"):
        get_writer_embedder("trained", cfg)


# -- calibration ------------------------------------------------------------------------------------------------------

CNN_CKPT = resolve_path("models/checkpoints/char_cnn.pt")


@pytest.mark.skipif(not CNN_CKPT.exists(), reason="needs the CNN recognizer checkpoint")
def test_calibration_scores_crops_and_reports_null_decoders(tmp_path):
    from src.evaluation import calibration
    from src.recognition.cnn_recognizer import CNNRecognizer

    build_fake_iam(tmp_path / "iam", n_writers=12, forms_per_writer=2, lines_per_form=3)
    cfg = json.loads(json.dumps(CFG))
    ds = load_iam_splits(unit="lines", config=cfg, root=tmp_path / "iam", splits_dir=tmp_path / "s")["test"]
    recs, agg = calibration.run_calibration(ds, CNNRecognizer(checkpoint_path=str(CNN_CKPT)), cfg, n_crops=4, seed=1)
    assert agg["n_scored"] + agg["n_failed"] == len(recs) <= 4
    assert 0 <= agg["cer_micro"] and agg["null_cer_random_letters"] > 0 and agg["n_reference_letters"] > 0
    again = calibration.select_crops(ds, 4, 1, cfg["evaluation"]["min_reference_letters"])
    assert again == calibration.select_crops(ds, 4, 1, cfg["evaluation"]["min_reference_letters"])   # deterministic
    assert {r["sample_id"] for r in recs} <= {s.sample_id for s in ds.samples}


# -- Task 4: official-count check ---------------------------------------------------------------------------------------

def test_check_split_expect_official_fails_loudly_on_a_non_official_corpus(tmp_path):
    build_fake_iam(tmp_path / "iam", n_writers=12, forms_per_writer=2, lines_per_form=3)
    r = subprocess.run([sys.executable, "scripts/check_split.py", "--root", str(tmp_path / "iam"),
                        "--splits-dir", str(tmp_path / "s"), "--unit", "lines", "--expect-official"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 1 and "MISMATCH" in r.stdout and "published 657" in r.stdout
    ok = subprocess.run([sys.executable, "scripts/check_split.py", "--root", str(tmp_path / "iam"),
                         "--splits-dir", str(tmp_path / "s"), "--unit", "lines"], cwd=REPO, capture_output=True, text=True)
    assert ok.returncode == 0                       # without the flag the same corpus is fine


# -- ROADMAP: test writers must never reach a training batch ----------------------------------------------

def test_sampler_refuses_test_writers_in_its_pool_and_batches():
    from src.data.leakage import WriterLeakageError, assert_no_test_writers
    with pytest.raises(WriterLeakageError):
        assert_no_test_writers(["001", "002"], {"002", "003"})
    assert_no_test_writers(["001"], {"002"})                       # disjoint -> silent
    from src.writer_id.data import WriterBatchSampler
    with pytest.raises(WriterLeakageError):
        WriterBatchSampler(None, {"w1": [1], "w2": [2]}, {"w1": 0, "w2": 1}, 2, 2, AUG, 0, forbidden_writers={"w2"})
