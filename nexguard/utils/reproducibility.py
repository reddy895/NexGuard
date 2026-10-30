"""
NexGuard — Reproducibility and Determinism Utilities
===================================================
Enforces deterministic random seed initializations across Python, NumPy,
and PyTorch (CPU and CUDA backends) to ensure scientific reproducibility.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Dict
import numpy as np

def seed_everything(seed: int = 42) -> int:
    """
    Sets global deterministic seeds across all random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed

def get_environment_signature() -> Dict[str, str]:
    """
    Captures platform, hardware, and library versions for experiment auditing.
    """
    sig = {
        "os": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
    }
    try:
        import torch
        sig["pytorch_version"] = torch.__version__
        sig["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            sig["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        sig["pytorch_version"] = "not_installed"

    try:
        import ultralytics
        sig["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        sig["ultralytics_version"] = "not_installed"

    return sig
