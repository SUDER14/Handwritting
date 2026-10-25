"""The by-name checkpoint shim must survive generate_fakes.py's own execution (Task 3.2) and replace its strict
positional load_checkpoint, which asserts len(old_model) == len(checkpoint) and fails on the released vatr.pth
(564 extra FID-InceptionV3 tensors)."""
import ast
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VATR = ROOT / "third_party" / "VATr"
CKPT = VATR / "files" / "vatr.pth"

sys.path.insert(0, str(ROOT / "scripts"))
import vatr_reproduce as vr  # noqa: E402

needs_vatr = pytest.mark.skipif(not CKPT.exists(), reason="released VATr checkpoint not present")


def test_script_defines_no_load_checkpoint_after_patch():
    code, ns = vr.patched_generate_fakes_code()
    tree = ast.parse((VATR / "generate_fakes.py").read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint" for n in tree.body)  # original has it
    names = [c.co_name for c in code.co_consts if hasattr(c, "co_name")]
    assert "load_checkpoint" not in names


@needs_vatr
def test_load_checkpoint_reached_in_script_namespace_passes_on_cpu():
    import torch
    from src.generator.vatr_backend import apply_runtime_shims, load_checkpoint_by_name
    apply_runtime_shims(VATR, "cpu")
    old = os.getcwd()
    os.chdir(VATR)
    try:
        # execute generate_fakes.py's module level (imports + defs) exactly as runpy would, stopping before __main__
        code, ns = vr.patched_generate_fakes_code()
        ns["__name__"] = "generate_fakes"                     # skip the `if __name__ == '__main__'` body
        exec(code, ns)
        assert ns["load_checkpoint"] is load_checkpoint_by_name   # survived the script's own definitions
        import generator as gen
        args = gen.FakeArgs()
        args.device, args.add_noise = "cpu", False
        model = gen.VATr(args)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        ck = ck["model"] if "model" in ck else ck
        n_ck, n_model = len(ck), len(model.state_dict())
        print(f"checkpoint tensors {n_ck}, model tensors {n_model}")
        assert n_ck != n_model
        orig = next(n for n in ast.parse((VATR / "generate_fakes.py").read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint")
        ons: dict = {}
        exec(compile(ast.Module([orig], []), "orig", "exec"), ons)
        with pytest.raises(AssertionError):                      # negative control: VATr's own loader fails here
            ons["load_checkpoint"](model, ck)
        ns["load_checkpoint"](model, ck)                         # the call generate_fakes makes; must not raise
    finally:
        os.chdir(old)
