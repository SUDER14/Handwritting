"""Writer-ID network: a small ResNet trained from scratch.

No pretrained weights: IAM line crops are single-channel, binarised, ~64 px tall
line art; ImageNet's RGB natural-image features are a poor prior for that and the
input stem would have to be re-initialised anyway. (Not A/B-tested here -- see the
run notes; an ImageNet-initialised variant is a possible later comparison.)

    1 x 64 x W  ->  stem  ->  4 stages of BasicBlocks  ->  global average pool
                ->  256-d embedding  ->  (training only) linear head over train-split writers

Global average pooling makes the encoder width-agnostic: it is trained on random
width windows and can embed a full line of any width.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.shortcut is None else self.shortcut(x)), inplace=True)


class WriterEncoder(nn.Module):
    """Image (B,1,H,W) -> embedding (B, widths[-1]). Works for any W >= ~32."""

    def __init__(self, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), stem_stride: int = 2):
        super().__init__()
        if len(widths) != len(blocks):
            raise ValueError("widths and blocks must have the same length")
        self.embedding_dim = int(widths[-1])
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, stem_stride, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(BasicBlock(cin, w, stride=(1 if i == 0 else 2) if j == 0 else 1))
                cin = w
        self.stages = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x)).mean(dim=(2, 3))     # global average pool -> (B, C)


class WriterIDNet(nn.Module):
    """Encoder + classification head over the training writers. Only the encoder is used at retrieval time."""

    def __init__(self, num_writers: int, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), stem_stride: int = 2):
        super().__init__()
        self.encoder = WriterEncoder(widths, blocks, stem_stride)
        self.classifier = nn.Linear(self.encoder.embedding_dim, num_writers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        return emb, self.classifier(emb)


def nt_xent_multi_positive(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent over a batch where every crop of the same writer is a positive for every other.

    For anchor i:  -mean_{p in P(i)} log( exp(s_ip / T) / sum_{a != i} exp(s_ia / T) ),  s = cosine similarity.
    With exactly two crops per writer this is the standard NT-Xent (SimCLR) loss; with K > 2 it is the
    multi-positive (supervised-contrastive) generalisation. Anchors with no positive in the batch are skipped.
    """
    z = F.normalize(embeddings, dim=1)
    sim = z @ z.T / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    per_anchor = -(log_prob.masked_fill(~pos, 0.0)).sum(1)[valid] / n_pos[valid]
    return per_anchor.mean()
