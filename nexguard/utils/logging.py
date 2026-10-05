"""
NexGuard — Structured Logging
==============================
Sets up a single consistent logger for the entire application.
Log level is controlled via NEXGUARD_LOG_LEVEL environment variable.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str = "nexguard") -> logging.Logger:
    """
    Returns a named logger configured for NexGuard.

    Logs are written to both stdout and logs/nexguard.log.
    The log level is read from the config module to avoid circular imports.
    """
    logger = logging.getLogger(name)

    # Only configure once
    if logger.handlers:
        return logger

    import os
    level_str = os.getenv("NEXGUARD_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    log_file = Path(__file__).resolve().parents[2] / "logs" / "nexguard.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning(f"Could not open log file {log_file}: {e}")

    # Suppress propagation to root logger
    logger.propagate = False

    return logger


# Module-level logger for use throughout the package
log = get_logger("nexguard")
