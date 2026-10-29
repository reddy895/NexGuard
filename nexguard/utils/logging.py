"""
NexGuard — Structured Logging
==============================
Single consistent logger for the entire application.
Log level is controlled via NEXGUARD_LOG_LEVEL environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def get_logger(name: str = "nexguard") -> logging.Logger:
    """
    Returns a named logger configured for NexGuard.

    Logs are written to stderr (console) and logs/nexguard.log.
    The log level is read from NEXGUARD_LOG_LEVEL env var.

    Args:
        name: Logger name, used to scope log messages.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_str = os.getenv("NEXGUARD_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_str, logging.WARNING)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = Path(__file__).resolve().parents[2] / "logs" / "nexguard.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", log_file, exc)

    logger.propagate = False
    return logger


# Module-level default logger
log = get_logger("nexguard")
