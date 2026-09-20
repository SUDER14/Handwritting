"""Configuration loading utilities.

Centralizes access to config/config.yaml so no module hard-codes tunable
parameters. Cached after first load.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and cache the YAML configuration file.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        Parsed configuration as a nested dict.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / relative_path
