"""Test-writer leakage guard. Every trained component (writer-ID net, recognizer, generator fine-tuning) must never see
a TEST-split writer. Trainers call `assert_no_test_writers` on the writers behind every training batch (and once on the
whole training set), so a config or split mistake fails loudly instead of quietly inflating a metric."""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import load_config, resolve_path


class WriterLeakageError(AssertionError):
    """A TEST-split writer reached a training pool or batch."""


def canonical_test_writers(config: dict | None = None, splits_dir: Path | None = None) -> frozenset[str]:
    """The canonical TEST-split writer ids (from the split JSON that `data.iam.split_source` selects)."""
    from src.data.iam import resolve_writer_splits, parse_forms
    cfg = config or load_config()
    iam = cfg["data"]["iam"]
    forms = parse_forms(resolve_path(iam["root"]) / "ascii" / "forms.txt")
    return frozenset(resolve_writer_splits(iam, sorted(set(forms.values())), splits_dir)["test"])


def assert_no_test_writers(writer_ids, forbidden, where: str = "training batch") -> None:
    leaked = sorted(set(map(str, writer_ids)) & set(map(str, forbidden)))
    if leaked:
        raise WriterLeakageError(f"test-split writer(s) {leaked[:10]} appear in a {where}")
