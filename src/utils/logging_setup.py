"""Shared logging configuration for the handwriting-ai project."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Logging level, defaults to INFO.

    Returns:
        Configured ``logging.Logger`` instance (idempotent — safe to call
        repeatedly without adding duplicate handlers).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
