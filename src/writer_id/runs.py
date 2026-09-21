"""Per-run checkpoint directories: models/checkpoints/<run_id>/, created once and never overwritten.

Contents of a run directory:
  config.json          resolved writer_id config + provenance (written first, so a crashed run is still identifiable)
  train_log.csv        one row per epoch
  best_epochNNN.pt     written whenever validation retrieval improves; NEVER replaced or deleted. The highest
                       NNN is the best checkpoint (each file is only written on improvement).
  final.pt             the last epoch's weights
  run.json             written at the end: best epoch, validation metrics, split fingerprints
  test_metrics.json    written once: retrieval on the TEST-split writers (unseen at training time)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.utils.config import resolve_path
from src.writer_id.model import WriterEncoder

CHECKPOINT_ROOT = "models/checkpoints"


def write_once(path: Path, text: str) -> None:
    """Create `path` with `text`; refuses to touch an existing file."""
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)


def new_run_dir(git_sha: str, dirty: bool, name: str | None = None, root: Path | None = None) -> tuple[str, Path]:
    root = Path(root) if root is not None else resolve_path(CHECKPOINT_ROOT)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"writerid_{stamp}_{git_sha}{'-dirty' if dirty else ''}" + (f"_{name}" if name else "")
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)        # a second run in the same second fails rather than sharing a dir
    return run_id, path


def is_writer_id_run(path: Path) -> bool:
    cfg = path / "config.json"
    if not cfg.exists():
        return False
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("kind") == "writer_id"
    except (OSError, ValueError):
        return False


def latest_run(root: Path | None = None, require_checkpoint: bool = True) -> Path | None:
    root = Path(root) if root is not None else resolve_path(CHECKPOINT_ROOT)
    if not root.exists():
        return None
    runs = sorted((p for p in root.iterdir() if p.is_dir() and is_writer_id_run(p)), key=lambda p: p.name)
    if require_checkpoint:
        runs = [p for p in runs if best_checkpoint(p) is not None]
    return runs[-1] if runs else None


def best_checkpoint(run_dir: Path) -> Path | None:
    files = [(int(m.group(1)), p) for p in Path(run_dir).glob("best_epoch*.pt") if (m := re.fullmatch(r"best_epoch(\d+)\.pt", p.name))]
    return max(files)[1] if files else None


def load_encoder(run_dir: Path, checkpoint: Path | None = None) -> tuple[WriterEncoder, dict]:
    """Rebuild the encoder of a run from its config.json and best (or given) checkpoint. Returns (encoder, run_config)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ckpt_path = checkpoint or best_checkpoint(run_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"{run_dir} has no best_epoch*.pt checkpoint")
    m = cfg["writer_id"]["model"]
    encoder = WriterEncoder(m["widths"], m["blocks"], m["stem_stride"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
    encoder.eval()
    return encoder, cfg


def read_test_metrics(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "test_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
