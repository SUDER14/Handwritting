"""CRNN-CTC line recognizer -- the LEGIBILITY INSTRUMENT (not a product; do not tune it for its own sake).

Input  : canonical writer-ID-style crop, uint8, INK HIGH (255 = ink), height 64, width variable
         (src/writer_id/data.py::canonical_from_iam_gray / canonical_from_binary).
Network: 5 conv blocks (3x3, BatchNorm, ReLU; 2x2 then 2x2 then 2x1 then 2x1 pooling) -> height 4 -> flatten ->
         Linear -> 2-layer bidirectional LSTM -> Linear(vocab + blank). One time step per 4 input columns.
Loss   : CTC (blank = 0). Decoding: greedy (argmax, collapse repeats, drop blanks).
Charset: built from the TRAIN-split transcriptions only; a character never seen in training decodes to nothing and
         counts as an error, which is what a fixed instrument should do.
"""
from __future__ import annotations

import editdistance
import numpy as np
import torch
import torch.nn as nn

BLANK = 0


def build_charset(texts) -> list[str]:
    return sorted({c for t in texts for c in t})


def encode(text: str, char_to_id: dict[str, int]) -> list[int]:
    return [char_to_id[c] for c in text if c in char_to_id]      # unseen chars are dropped from the target


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, charset: list[str]) -> list[str]:
    """logits (T, B, C) -> strings."""
    best = logits.argmax(-1).T.cpu().numpy()                     # (B, T)
    out = []
    for row, n in zip(best, lengths.tolist()):
        prev, chars = BLANK, []
        for k in row[:n]:
            if k != prev and k != BLANK:
                chars.append(charset[k - 1])
            prev = k
        out.append("".join(chars))
    return out


def cer_wer(refs: list[str], hyps: list[str]) -> tuple[float, float]:
    """Corpus CER and WER (total edit distance / total reference length)."""
    ce = sum(editdistance.eval(r, h) for r, h in zip(refs, hyps))
    cn = sum(len(r) for r in refs)
    we = sum(editdistance.eval(r.split(), h.split()) for r, h in zip(refs, hyps))
    wn = sum(len(r.split()) for r in refs)
    return ce / max(cn, 1), we / max(wn, 1)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, num_chars: int, hidden: int = 256, widths=(32, 64, 128, 128, 192)):
        super().__init__()
        w = widths
        self.cnn = nn.Sequential(
            _block(1, w[0]), nn.MaxPool2d(2, 2),                 # 32 x W/2
            _block(w[0], w[1]), nn.MaxPool2d(2, 2),              # 16 x W/4
            _block(w[1], w[2]), _block(w[2], w[3]), nn.MaxPool2d((2, 1), (2, 1)),   # 8 x W/4
            _block(w[3], w[4]), nn.MaxPool2d((2, 1), (2, 1)),    # 4 x W/4
        )
        self.proj = nn.Linear(w[4] * 4, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(2 * hidden, num_chars + 1)          # + CTC blank at index 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 1, 64, W) float in [0, 1] -> log-probs (T, B, C+1) with T = W // 4."""
        f = self.cnn(x)                                          # (B, C, 4, W/4)
        b, c, h, t = f.shape
        f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
        f = torch.relu(self.proj(f))
        f, _ = self.rnn(f)
        return self.out(f).log_softmax(-1)


def pad_batch(crops: list[np.ndarray], multiple: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """List of (64, w) uint8 crops -> (B, 1, 64, W) float tensor (paper = 0) and per-item output lengths."""
    W = max(c.shape[1] for c in crops)
    W = -(-W // multiple) * multiple
    x = np.zeros((len(crops), 1, crops[0].shape[0], W), np.float32)
    for i, c in enumerate(crops):
        x[i, 0, :, : c.shape[1]] = c.astype(np.float32) / 255.0
    lengths = torch.tensor([max(1, c.shape[1] // 4) for c in crops], dtype=torch.long)
    return torch.from_numpy(x), lengths


class Recognizer:
    """Frozen inference wrapper: `transcribe(list of canonical crops) -> list[str]`."""

    def __init__(self, model: CRNN, charset: list[str]):
        self.model, self.charset = model.eval(), charset

    @torch.inference_mode()
    def transcribe(self, crops: list[np.ndarray], batch_size: int = 16) -> list[str]:
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        out = [""] * len(crops)
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            x, lengths = pad_batch([crops[i] for i in idx])
            for i, text in zip(idx, greedy_decode(self.model(x), lengths, self.charset)):
                out[i] = text
        return out
