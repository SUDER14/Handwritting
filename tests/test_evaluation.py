"""Tests for the evaluation harness: CER, writer-embedder interface, leave-one-out leak, persistence."""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.evaluation import harness
from src.evaluation import loo as loo_module
from src.evaluation.loo import leave_one_out_ssim, style_profile_without
from src.evaluation.text_metrics import cer, levenshtein
from src.evaluation.writer_id import RuleBasedStubEmbedder, WriterEmbedder, cosine_distance, get_writer_embedder
from src.features.style_extractor import StyleFeatureExtractor
from src.generator.baseline_generator import BaselineGlyphGenerator
from src.segmentation.segmenter import Glyph
from src.utils.config import load_config, resolve_path


# -- text metrics -------------------------------------------------------------

def test_levenshtein_known_values():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3 and levenshtein("abc", "") == 3 and levenshtein("abc", "abc") == 0


def test_cer():
    assert cer("quick", "quick") == 0.0
    assert cer("quick", "quack") == pytest.approx(0.2)
    assert cer("ab", "abcdef") == pytest.approx(2.0)      # insertions can push CER above 1
    with pytest.raises(ValueError):
        cer("", "x")


# -- writer-ID interface ------------------------------------------------------

def test_writer_embedder_is_abstract_and_stub_is_flagged():
    with pytest.raises(TypeError):
        WriterEmbedder()  # abstract
    stub = get_writer_embedder("stub")
    assert isinstance(stub, RuleBasedStubEmbedder) and stub.is_stub is True
    assert stub.describe() == {"name": "stub_rule_based_v0", "is_stub": True}
    with pytest.raises(NotImplementedError):
        get_writer_embedder("trained_writer_id_v1")


def test_cosine_distance():
    assert cosine_distance(np.array([1.0, 0]), np.array([2.0, 0])) == pytest.approx(0.0)
    assert cosine_distance(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine_distance(np.zeros(3), np.ones(3))


def test_stub_embedder_is_scale_free_and_self_distance_zero():
    img = np.zeros((80, 300), np.uint8)
    cv2.putText(img, "hello", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 3)
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    stub = get_writer_embedder("stub")
    assert stub.distance(img, img) == pytest.approx(0.0, abs=1e-12)
    assert stub.embed(img).shape == stub.embed(big).shape == (6,)
    with pytest.raises(ValueError):
        stub.embed(np.zeros((50, 50), np.uint8))          # blank image: no invented embedding


# -- leave-one-out must not see the held-out crop ------------------------------

def _glyph_image(h: int, w: int, thickness: int) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    img[:, (w - thickness) // 2:(w + thickness) // 2] = 255
    img[h // 2 - thickness // 2:h // 2 + thickness // 2 + 1, :] = 255
    return img


def make_analysis(q_image: np.ndarray | None = None) -> SimpleNamespace:
    """A fake 5-letter sample 'quick' where 'q' is deliberately extreme (huge and thick), so any
    influence of the held-out crop on the profile is impossible to miss."""
    q_image = _glyph_image(200, 150, 40) if q_image is None else q_image
    specs = [("q", q_image, 10), ("u", _glyph_image(60, 40, 6), 200), ("i", _glyph_image(62, 30, 6), 260),
             ("c", _glyph_image(58, 42, 6), 320), ("k", _glyph_image(64, 44, 6), 390)]
    glyphs = [Glyph(index=i, bbox=(x, 300 - img.shape[0], img.shape[1], img.shape[0]), image=img,
                    centroid=(x + img.shape[1] / 2, 0.0)) for i, (_, img, x) in enumerate(specs)]
    chars = [c for c, _, _ in specs]
    return SimpleNamespace(
        chars=chars, observed_glyphs={c: g.image for c, g in zip(chars, glyphs)},
        segmentation=SimpleNamespace(glyphs=glyphs, baseline_y=300, x_height_top_y=240),
    )


def _profile_fields(p) -> dict:
    d = {k: getattr(p, k) for k in (
        "mean_height_px", "mean_width_px", "mean_aspect_ratio", "mean_slant_deg", "slant_std_deg",
        "mean_stroke_width_px", "mean_curvature_score", "mean_char_spacing_px", "baseline_y", "x_height_top_y",
        "x_height_px")}
    d["feature_vector"] = p.feature_vector.tolist()
    d["glyph_chars"] = [f.char for f in p.glyph_features]
    return d


def test_profile_excluding_equals_profile_of_the_rest_only():
    a = make_analysis()
    held = style_profile_without(a, "q")
    rest = [g for c, g in zip(a.chars, a.segmentation.glyphs) if c != "q"]
    baseline = int(np.median([g.bbox[1] + g.bbox[3] for g in rest]))
    top = int(np.median([g.bbox[1] for g in rest]))
    expected = StyleFeatureExtractor().extract_style_profile(["u", "i", "c", "k"], rest, baseline, top)
    assert _profile_fields(held) == _profile_fields(expected)
    assert "q" not in [f.char for f in held.glyph_features]
    assert held.mean_height_px < 70            # the 200-px 'q' is not in the average


def test_held_out_crop_is_unreachable_from_the_scoring_profile():
    """Change ONLY the held-out glyph's pixels/box; the profile used to score it must not change at all.
    (This is the property the old code violated.)"""
    a1 = make_analysis(_glyph_image(200, 150, 40))
    a2 = make_analysis(_glyph_image(20, 12, 2))            # wildly different 'q'
    assert _profile_fields(style_profile_without(a1, "q")) == _profile_fields(style_profile_without(a2, "q"))

    # ...and the test has teeth: the FULL-sample profile (what the old baseline path used) does depend on it.
    full = lambda a: StyleFeatureExtractor().extract_style_profile(  # noqa: E731
        a.chars, a.segmentation.glyphs, a.segmentation.baseline_y, a.segmentation.x_height_top_y)
    assert _profile_fields(full(a1)) != _profile_fields(full(a2))


def test_spacing_ignores_the_gap_that_straddles_the_held_out_glyph():
    a = make_analysis()
    # Hold out 'i' (middle). Gaps u-i and i-c touch it; only q-u and c-k may count.
    g = a.segmentation.glyphs
    gap = lambda i: g[i + 1].bbox[0] - (g[i].bbox[0] + g[i].bbox[2])  # noqa: E731
    expected = np.mean([gap(0), gap(3)])
    assert style_profile_without(a, "i").mean_char_spacing_px == pytest.approx(expected)


def test_every_occurrence_of_the_held_out_letter_is_removed():
    a = make_analysis()
    a.chars = ["q", "u", "q", "c", "k"]                     # 'q' occurs twice; observed_glyphs keeps one
    assert [f.char for f in style_profile_without(a, "q").glyph_features] == ["u", "c", "k"]


def test_leave_one_out_scoring_path_uses_the_leak_free_profile(monkeypatch):
    """End to end through leave_one_out_ssim: the style handed to the generator when scoring 'q' must be
    identical no matter what the real 'q' looks like, and must equal the rest-only profile."""
    seen: dict[str, list] = {}
    real = BaselineGlyphGenerator.generate_unobserved

    def spy(self, char, style):
        seen.setdefault(char, []).append(_profile_fields(style))
        return real(self, char, style)

    monkeypatch.setattr(BaselineGlyphGenerator, "generate_unobserved", spy)
    cfg = load_config()
    s1 = leave_one_out_ssim(make_analysis(_glyph_image(200, 150, 40)), cfg)
    s2 = leave_one_out_ssim(make_analysis(_glyph_image(20, 12, 2)), cfg)
    assert set(s1) == set("quick") == set(s2)
    assert seen["q"][0] == seen["q"][1]                     # identical style regardless of the held-out crop
    assert seen["q"][0]["glyph_chars"] == ["u", "i", "c", "k"]
    assert loo_module.style_profile_without(make_analysis(), "q").mean_height_px == seen["q"][0]["mean_height_px"]


def test_leave_one_out_needs_two_distinct_letters():
    a = make_analysis()
    a.chars = ["q"] * 5
    a.observed_glyphs = {"q": a.observed_glyphs["q"]}
    assert leave_one_out_ssim(a, load_config()) == {}


# -- persistence --------------------------------------------------------------

def _fake_record(is_smoke=False, split="val"):
    sample = {"writer_id": "003", "reference_id": "r", "target_id": "t", "error": None, "cer": 0.5,
              "edit_distance": 3, "n_target_letters": 6, "n_glyphs_detected": 6,
              "style_distance_reference": 0.01, "style_distance_heldout": 0.02, "loo_ssim_mean": 0.2}
    return {
        "schema_version": 1, "git": {"sha": "abc12345", "dirty": False, "branch": "master"}, "split": split,
        "unit": "lines", "backend": "baseline", "is_smoke_test": is_smoke,
        "writer_embedder": {"name": "stub_rule_based_v0", "is_stub": True}, "cli_args": {}, "resolved_config": load_config(),
        "checkpoints": {"cnn": {"path": "c.pt", "exists": True, "sha256": "aa"}, "vae": None},
        "split_files": None, "skipped_writers": [], "samples": [sample], "aggregate": harness.aggregate([sample]),
    }


def test_write_run_persists_json_and_appends_flat_index(tmp_path):
    p1 = harness.write_run(tmp_path, _fake_record())
    p2 = harness.write_run(tmp_path, _fake_record(split="test"))
    assert p1 != p2 and p1.parent.name.endswith("_abc12345") and p1.name == "metrics.json"

    m = json.loads(p1.read_text())
    for key in ("run_id", "git", "resolved_config", "checkpoints", "split", "samples", "aggregate", "timestamp_utc"):
        assert key in m
    assert m["git"]["sha"] == "abc12345" and m["resolved_config"]["evaluation"]["targets_per_writer"] >= 1
    assert m["run_id"] == p1.parent.name

    rows = list(csv.DictReader((tmp_path / "index.csv").open()))
    assert len(rows) == 2 and list(rows[0].keys()) == harness.INDEX_COLUMNS
    assert rows[0]["split"] == "val" and rows[1]["split"] == "test"
    assert float(rows[0]["cer_micro"]) == pytest.approx(0.5) and rows[0]["embedder_is_stub"] == "True"


def test_index_schema_mismatch_is_refused(tmp_path):
    (tmp_path / "index.csv").write_text("run_id,something_else\n1,2\n")
    with pytest.raises(ValueError, match="columns"):
        harness.write_run(tmp_path, _fake_record())


def test_aggregate_refuses_to_report_a_single_metric():
    good = _fake_record()["samples"][0]
    with pytest.raises(RuntimeError, match="headline"):
        harness.aggregate([{**good, "style_distance_reference": None}])
    with pytest.raises(RuntimeError, match="nothing to aggregate"):
        harness.aggregate([{"error": "boom", "target_id": "x"}])
    agg = harness.aggregate([good, {"error": "boom", "target_id": "y", "writer_id": "9"}])
    assert agg["n_samples"] == 1 and agg["n_failed"] == 1     # failures are counted, not hidden


def test_write_run_rejects_non_finite_numbers(tmp_path):
    rec = _fake_record()
    rec["aggregate"]["cer_mean"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        harness.write_run(tmp_path, rec)


def test_truncate_and_clean_text():
    assert harness.render_text_clean("Mr. Gaitskell, 1st  time!") == "mr gaitskell st time"
    assert harness.truncate_text("a move to stop mr gaitskell", 12) == "a move to"
    assert harness.letters_only("A MOVE, to 3") == "amoveto"


# -- the whole harness on a fake IAM tree (baseline backend) --------------------

CNN_CKPT = resolve_path("models/checkpoints/char_cnn.pt")


@pytest.mark.skipif(not CNN_CKPT.exists(), reason="CNN recognizer checkpoint needed for legibility (CER)")
def test_harness_runs_end_to_end_on_a_fake_iam_split(tmp_path):
    from src.data.iam import load_iam_splits
    from src.evaluation.writer_id import get_writer_embedder
    from src.recognition.cnn_recognizer import CNNRecognizer
    from tests.iam_fixture import build_fake_iam

    build_fake_iam(tmp_path / "iam", n_writers=12, forms_per_writer=2, lines_per_form=3)
    cfg = json.loads(json.dumps(load_config()))
    splits = load_iam_splits(unit="lines", config=cfg, root=tmp_path / "iam", splits_dir=tmp_path / "splits")
    ctx = harness.EvalContext(config=cfg, backend="baseline", recognizer=CNNRecognizer(checkpoint_path=str(CNN_CKPT)),
                              embedder=get_writer_embedder("stub", cfg))
    records, skipped = harness.run_iam_split(ctx, splits["val"], max_writers=2)
    assert records, skipped
    ok = [r for r in records if r["error"] is None]
    assert ok, [r["error"] for r in records]
    # Every successful sample carries BOTH headline metrics, and only val-split writers were used.
    for r in ok:
        assert r["cer"] is not None and r["style_distance_reference"] is not None
        assert r["writer_id"] in splits["val"].writer_ids
    agg = harness.aggregate(records)
    assert agg["cer_micro"] >= 0 and agg["style_distance_reference_mean"] is not None
