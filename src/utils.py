"""
NexGuard Utility Module
Provides hardware detection, file path validation, directory setup, and formatting utilities.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional


def detect_device() -> str:
    """
    Detects hardware accelerator availability (CUDA vs CPU).
    Returns 'cuda' or 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_device_info() -> Tuple[str, str]:
    """
    Returns a tuple of (device_type, device_description).
    Example: ('cuda', 'NVIDIA GeForce RTX 3080') or ('cpu', 'CPU (Host)')
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA GPU"
            return "CUDA", name
    except Exception:
        pass
    return "CPU", "Host CPU Processor"


def validate_file_path(path_str: str, allowed_extensions: Optional[Tuple[str, ...]] = None) -> bool:
    """
    Validates that a file path exists and is a regular file.
    Optionally checks against allowed file extensions (case-insensitive).
    """
    if not path_str or not isinstance(path_str, str):
        return False

    path = Path(path_str.strip())
    if not path.is_file():
        return False

    if allowed_extensions:
        exts = tuple(ext.lower() for ext in allowed_extensions)
        if path.suffix.lower() not in exts:
            return False

    return True


def ensure_dir(directory: str) -> Path:
    """Ensures that a directory path exists."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_snapshot_filename(prefix: str = "detection") -> str:
    """Generates a timestamped filename for evidence snapshots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.jpg"
