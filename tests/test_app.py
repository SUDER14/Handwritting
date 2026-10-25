"""Headless smoke test of the Streamlit app (ROADMAP GATE 6: launches without errors) via streamlit.testing.AppTest."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import app_logic as L  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def test_vatr_status_names_what_is_missing():
    ok, msg = L.vatr_status({"vatr": {"root": "third_party/__no_such_dir__", "checkpoint": "files/vatr.pth"}})
    assert not ok and "checkpoint is missing" in msg


def test_app_launches_and_vatr_switch_is_disabled_with_a_reason_when_unavailable():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    ok, msg = L.vatr_status()
    toggle = at.toggle(key="use_vatr")
    assert toggle.disabled == (not ok)
    if not ok:
        assert any("VATr is disabled" in w.value for w in at.warning)
    assert [r.label for r in at.radio if r.key == "backend"] == ["Backend"]
    assert list(at.radio(key="backend").options) == ["real", "baseline", "glyph_vae"]


def test_upload_mode_without_a_file_warns_instead_of_crashing():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="source").set_value("Upload my own line").run()
    at.button(key="generate").click().run()
    assert not at.exception
    assert any("reference line" in w.value for w in at.warning)


def test_demo_writer_baseline_generation_end_to_end_shows_scores():
    """Full path on the IAM demo data: generate with the baseline and score with the frozen instruments."""
    from src.utils.config import resolve_path
    if not resolve_path("data/iam").exists() or not any(resolve_path("data/iam").iterdir()):
        pytest.skip("data/iam not materialised on this machine")
    at = AppTest.from_file(APP, default_timeout=600).run()
    at.radio(key="backend").set_value("baseline")
    at.text_input(key="text").set_value("the quick fox")
    at.button(key="generate").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    labels = [m.label for m in at.metric]
    assert any("CER" in l for l in labels) and any("HWD" in l for l in labels) and any("writer-ID" in l for l in labels)
    assert all(float(m.value) >= 0 for m in at.metric)
