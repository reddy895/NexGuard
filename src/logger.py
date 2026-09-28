"""
NexGuard Logging Module
Provides structured application logging for console and file output.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "nexguard.log"

_LOGGER = None


def setup_logger(name: str = "NexGuard", log_file: Path = LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the global NexGuard logger."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[WARNING] Failed to initialize file logger at {log_file}: {e}", file=sys.stderr)

    # Console Handler (clean stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _LOGGER = logger
    return logger


def get_logger() -> logging.Logger:
    """Returns the logger instance, initializing it if necessary."""
    if _LOGGER is None:
        return setup_logger()
    return _LOGGER
