"""Non-UI logic of the Streamlit app (app/app.py), kept importable and testable without Streamlit.

Backends: real / baseline / glyph_vae (CPU) and vatr (needs the released checkpoint AND CUDA). Scores shown next to an
output are the SAME frozen instruments as the evaluation protocol: legibility = CER of the pinned CRNN reading the output,
fidelity = pinned writer-ID cosine distance and official HWD against the writer's real lines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation import protocol as P
from src.evaluation.harness import render_text_clean, truncate_text
from src.evaluation.text_metrics import levenshtein
from src.utils.config import load_config, resolve_path

BACKENDS = ["real", "baseline", "glyph_vae"]


def vatr_status(cfg: dict | None = None) -> tuple[bool, str]:
    """(available, message). VATr needs the released checkpoint and a CUDA GPU (CPU is ~35 s per line)."""
    cfg = cfg or load_config()
    vc = cfg.get("vatr", {})
    ckpt = resolve_path(vc.get("root", "third_party/VATr")) / vc.get("checkpoint", "files/vatr.pth")
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:                                                   # noqa: BLE001
        has_cuda = False
    problems = []
    if not ckpt.exists():
        problems.append(f"the released VATr checkpoint is missing ({ckpt.as_posix()})")
    if not has_cuda:
        problems.append("no CUDA GPU is available")
    if problems:
        return False, "VATr is disabled: " + " and ".join(problems) + ". See notebooks/colab_train.ipynb to run it on Colab."
    return True, "VATr is available (released checkpoint + CUDA)."


@dataclass
class DemoCase:
    writer: str
    ref_gray: np.ndarray
    ref_text: str
    heldout: list[np.ndarray] = field(default_factory=list)       # the writer's real held-out target lines
    heldout_texts: list[str] = field(default_factory=list)


class Instruments:
    """The frozen scoring instruments, built lazily (CRNN + writer-ID net + the 500 MB HWD backbone)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._scorer = None

    @property
    def scorer(self) -> P.Scorer:
        if self._scorer is None:
            from src.evaluation.hwd_metric import HWDFeatures
            from src.evaluation.writer_id import get_writer_embedder
            from src.recognition.crnn import load_crnn
            ev = self.cfg["evaluation"]
            embedder = get_writer_embedder("trained", self.cfg, run_dir=None)
            self._scorer = P.Scorer(load_crnn(resolve_path(ev["crnn_run"])), embedder, HWDFeatures(),
                                    self.cfg["crnn"]["input_height"])
        return self._scorer

    def score(self, output: np.ndarray, text: str, real_lines: list[np.ndarray]) -> dict:
        """{cer, hyp, wid_dist, hwd, n_real}: output line vs the text it should show and vs the writer's real lines."""
        from src.evaluation.hwd_metric import hwd, set_mean
        s = self.scorer
        hyp = render_text_clean(s.read([output])[0])
        ref = render_text_clean(text)
        cer = levenshtein(ref, hyp) / max(len(ref), 1)
        e = s.embed(output)
        r = [s.embed(g) for g in real_lines]
        unit = lambda v: v / np.linalg.norm(v)                                          # noqa: E731
        m = np.mean([unit(v) for v in r], axis=0)
        wid = float(1.0 - np.dot(unit(e), m) / np.linalg.norm(m))
        h = hwd(set_mean([s.hwd_entry(output)]), set_mean([s.hwd_entry(g) for g in real_lines]))
        return {"cer": float(cer), "hyp": hyp, "wid_dist": wid, "hwd": float(h), "n_real": len(real_lines)}


def demo_writers(cfg: dict | None = None) -> list[str]:
    """The seeded 20-writer test subset of the protocol (needs data/iam)."""
    from src.data.iam import load_iam
    cfg = cfg or load_config()
    ev, pc = cfg["evaluation"], cfg["evaluation"]["protocol"]
    ds = load_iam("test", unit="lines")
    return P.choose_writers(ds, pc["n_writers"], pc["seed"], max(pc["ks"]) + pc["targets_per_writer"] // 2 + 1)


def demo_case(writer: str, cfg: dict | None = None, K: int = 1) -> DemoCase:
    from src.data.iam import load_iam
    cfg = cfg or load_config()
    ev, pc = cfg["evaluation"], cfg["evaluation"]["protocol"]
    ds = load_iam("test", unit="lines")
    samples = [s for s in ds.samples if s.writer_id == writer]
    index = {s.sample_id: i for i, s in enumerate(ds.samples)}
    refs, targets = P.split_lines(samples, K, pc["targets_per_writer"], ev["min_reference_letters"], pc["seed"], writer)
    r = refs[0]
    return DemoCase(writer=writer, ref_gray=ds.read_native(index[r.sample_id]), ref_text=r.transcription,
                    heldout=[ds.read_native(index[t.sample_id]) for t in targets],
                    heldout_texts=[truncate_text(t.transcription, ev["max_target_chars"]) for t in targets])


def make_backend(name: str, cfg: dict | None = None):
    cfg = cfg or load_config()
    from src.evaluation import backends as B
    if name == "baseline":
        return B.BaselineBackend(cfg)
    if name == "glyph_vae":
        return B.NeuralBackend(cfg, str(resolve_path(cfg["training"]["checkpoint_path"])))
    if name == "vatr":
        ok, msg = vatr_status(cfg)
        if not ok:
            raise RuntimeError(msg)
        from src.generator.vatr_backend import VATrBackend
        return VATrBackend(cfg)
    raise ValueError(f"unknown backend {name!r}")


def generate_line(name: str, ref_gray: np.ndarray, ref_text: str, text: str, cfg: dict | None = None) -> np.ndarray | None:
    """One reference line + target text -> uint8 gray line (dark ink on light paper), via the protocol's backend interface."""
    text = render_text_clean(text)
    if not text:
        return None
    return make_backend(name, cfg).generate([P.Ref(ref_gray, ref_text)], text)
