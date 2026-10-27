"""Device-mismatch class on the protocol path (Colab: "Input type (torch.FloatTensor) and weight type
(torch.cuda.FloatTensor) should be the same"). No GPU here, so every instrument is moved to the 'meta' device and
guarded: a forward whose input tensor is not on its weights' device raises the same RuntimeError CUDA would. The guard
swaps the (shape-only) meta output for a CPU tensor so the rest of the pipeline can run on one writer."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torchvision import transforms as T

from src.evaluation import hwd_metric
from src.evaluation import protocol as P
from src.evaluation.writer_id import TrainedWriterEmbedder
from src.recognition.crnn import CRNN, Recognizer
from src.writer_id.model import WriterEncoder

CHARSET = list("abcdefghijklmnopqrstuvwxyz ")
TEXTS = ["the quick brown fox jumps", "over the lazy dog again", "pack my box with five dozen",
         "liquor jugs sphinx of black quartz", "how vexingly quick daft zebras", "waltz bad nymph for quick"]


def guard(module: nn.Module, device: str = "meta") -> nn.Module:
    """Put `module` on `device` and make a wrong-device input fail like CUDA does; outputs come back on the CPU."""
    module.to(device).eval()

    def check(mod, args):
        dev = next(mod.parameters()).device
        for a in args:
            if isinstance(a, torch.Tensor) and a.device != dev:
                raise RuntimeError(f"Input type ({a.device}) and weight type ({dev}) should be the same")

    def to_cpu(mod, args, out):
        g = torch.Generator().manual_seed(0)
        return torch.rand(out.shape, generator=g) + 0.1

    module.register_forward_pre_hook(check)
    module.register_forward_hook(to_cpu)
    return module


class _Tokens(nn.Module):                    # stands in for the official AdjustDims: (B, C, H, W) -> (B, H*W, C)
    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


class _StubHWDScore:
    """Same device behaviour as the official HWDScore: the backbone is moved to the compute device at construction."""

    def __init__(self, height=32):
        self.backbone = SimpleNamespace(model=nn.Sequential(nn.Conv2d(3, 8, 3), _Tokens()))
        guard(self.backbone.model)
        self.transforms = T.Compose([T.Resize((height, 64)), T.ToTensor()])


class _DS:
    samples = [SimpleNamespace(sample_id=f"007-{i:02d}", writer_id="007", transcription=t) for i, t in enumerate(TEXTS)]

    def read_native(self, i):
        img = np.full((64, 200), 255, np.uint8)
        img[20:40, 20:180] = 0
        return img


class _Backend:
    name = "fake"

    def generate(self, refs, text):
        img = np.full((64, 100), 255, np.uint8)
        img[20:40, 10:90] = 0
        return img


def _scorer(monkeypatch):
    monkeypatch.setattr(hwd_metric, "_official", lambda: _StubHWDScore)
    recognizer = Recognizer(guard(CRNN(len(CHARSET), hidden=16, widths=(4, 4, 4, 4, 4))), CHARSET)
    embedder = object.__new__(TrainedWriterEmbedder)             # real embed(); only the weights are random
    embedder.encoder = guard(WriterEncoder((4, 4, 4, 4), (1, 1, 1, 1)))
    embedder._height = 64
    embedder._eval_cfg = {"mode": "windows", "window_width": 64, "window_stride": 32, "max_windows": 4}
    return P.Scorer(recognizer, embedder, hwd_metric.HWDFeatures(), 64)


CFG = {"targets_per_writer": 4, "min_reference_letters": 12, "max_target_chars": 40}


@pytest.mark.parametrize("backend", [None, _Backend()], ids=["real_row", "backend_row"])
def test_one_writer_scores_with_every_instrument_on_a_non_cpu_device(monkeypatch, backend):
    row = P.run_writer(_scorer(monkeypatch), backend, _DS(), _DS.samples, 2, CFG, 1337, "007")
    assert row is not None and row["all"]["n_lines"] >= 1
    assert np.isfinite(row["all"]["hwd"]) and np.isfinite(row["all"]["wid_dist"])


def test_vatr_backend_sends_style_and_text_tensors_to_its_device():
    from src.generator.vatr_backend import NUM_STYLE, STYLE_H, STYLE_MAX_W, VATrBackend
    seen = {}

    class Converter:
        def encode(self, words):
            return torch.ones(sum(map(len, words)), dtype=torch.int), torch.tensor([len(w) for w in words]), None

    class Model:
        netconverter = Converter()

        def _generate_fakes(self, style, enc, lens, pos):
            seen["style"], seen["enc"] = style.device, enc.device
            return [np.zeros((STYLE_H, 8), np.float32) for _ in lens]

    b = object.__new__(VATrBackend)
    b.model, b.device, b._torch, b._cached = Model(), "meta", torch, None
    b.alphabet, b.dropped_chars = set("abcdefghijklmnopqrstuvwxyz"), {}
    b._style = lambda refs: torch.zeros(1, NUM_STYLE, STYLE_H, STYLE_MAX_W).to(b.device)
    assert b.generate([], "the fox") is not None
    assert seen == {"style": torch.device("meta"), "enc": torch.device("meta")}
