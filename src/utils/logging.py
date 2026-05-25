"""Console + file logger. Course rubric requires text logs of training output."""

from __future__ import annotations

import logging
from pathlib import Path


def build_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to both stdout and ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file.setFormatter(fmt)
    logger.addHandler(file)

    return logger
