"""The runtime wandb stub must not break `importlib.util.find_spec("wandb")` (accelerate/transformers call it when HWD
imports them) and must be removed before scoring."""
import importlib.util
import sys

import pytest

from src.generator.vatr_backend import install_wandb_stub, remove_wandb_stub


@pytest.fixture
def no_wandb(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    yield
    sys.modules.pop("wandb", None)


def test_stub_has_spec_so_find_spec_does_not_raise(no_wandb):
    install_wandb_stub()
    assert sys.modules["wandb"].__spec__ is not None
    importlib.util.find_spec("wandb")                    # raised ValueError("wandb.__spec__ is None") before the fix


def test_remove_drops_only_the_stub(no_wandb):
    install_wandb_stub()
    remove_wandb_stub()
    assert "wandb" not in sys.modules
    import types
    real = types.ModuleType("wandb")                     # stands in for a genuinely imported wandb
    sys.modules["wandb"] = real
    remove_wandb_stub()
    assert sys.modules["wandb"] is real


def test_accelerate_and_transformers_import_with_stub_installed(no_wandb):
    pytest.importorskip("accelerate")
    pytest.importorskip("transformers")
    install_wandb_stub()
    import accelerate  # noqa: F401
    from transformers import AutoModel  # noqa: F401


def test_check_complete_refuses_partial_and_accepts_full(tmp_path, capsys):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import vatr_reproduce as vr
    for d, n in (("Fake_test", 3), ("Real_test", 3)):
        (tmp_path / d / "001").mkdir(parents=True)
        for i in range(n):
            (tmp_path / d / "001" / f"{i:05d}.png").write_bytes(b"x")
    vr.check_complete(tmp_path, 3)                       # full: no raise
    assert "expected 3 images" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="partial"):
        vr.check_complete(tmp_path, 4)
    (tmp_path / "Fake_test" / "001" / "00002.png").unlink()
    with pytest.raises(SystemExit, match="partial"):
        vr.check_complete(tmp_path, 3)
