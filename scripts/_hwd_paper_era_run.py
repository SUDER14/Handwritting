"""Subprocess helper for vatr_hwd_audit.py: run the REAL HWD code at the paper-era commit (git worktree) on two folders.
usage: _hwd_paper_era_run.py <worktree> <fake_dir> <real_dir>   -> last stdout line is a JSON dict.
The worktree's top-level packages are `metrics` and `datasets` (the latter clashes with HuggingFace `datasets`), so they
are registered by path with empty __init__ semantics; only the modules HWDScore needs are imported."""
import json
import sys
import types
from pathlib import Path

wt = Path(sys.argv[1])
sys.path.insert(0, str(wt))
for name in ("metrics", "datasets"):
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(wt / name)]
    sys.modules[name] = pkg

from metrics.scores import HWDScore            # noqa: E402  (worktree code, not the pinned package)
from datasets.folder_dataset import FolderDataset  # noqa: E402

fake = FolderDataset(sys.argv[2], extension="png")
real = FolderDataset(sys.argv[3], extension="png")
score = HWDScore()
pooled = float(score(real, fake))               # README usage: score(real_dataset, fake_dataset); pooled over the whole set
print(json.dumps({"pooled": pooled, "n_fake": len(fake), "n_real": len(real), "worktree": str(wt)}))
