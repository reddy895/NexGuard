"""
NexGuard — Device and Compute Acceleration Manager
==================================================
Manages device resolution (CUDA, CPU, MPS) with fail-safe fallback:
- Queries torch.cuda capability and memory allocation
- Detects GPU availability, driver status, and compute limits
- Provides non-crashing CPU fallback if CUDA initialization fails
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.device")

@dataclass
class DeviceInfo:
    device_string: str
    device_type: str  # 'cuda', 'cpu', 'mps'
    device_name: str
    vram_total_mb: float = 0.0
    vram_available_mb: float = 0.0
    cuda_available: bool = False
    amp_supported: bool = False

class DeviceManager:
    """
    Resolves and monitors hardware devices for training and inference.
    """

    @classmethod
    def resolve_device(cls, requested: str = "auto") -> DeviceInfo:
        req = requested.strip().lower()

        # Check CUDA
        cuda_ok = False
        dev_name = "CPU"
        vram_total = 0.0
        vram_free = 0.0

        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                dev_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_total = props.total_memory / (1024 * 1024)
                vram_free = (props.total_memory - torch.cuda.memory_allocated(0)) / (1024 * 1024)
        except Exception as exc:
            log.warning("PyTorch CUDA query failed: %s", exc)
            cuda_ok = False

        if req == "cuda":
            if cuda_ok:
                return DeviceInfo(
                    device_string="cuda:0",
                    device_type="cuda",
                    device_name=dev_name,
                    vram_total_mb=vram_total,
                    vram_available_mb=vram_free,
                    cuda_available=True,
                    amp_supported=True,
                )
            else:
                log.warning("CUDA requested but unavailable. Falling back safely to CPU.")
                return DeviceInfo(
                    device_string="cpu",
                    device_type="cpu",
                    device_name="Host CPU (CUDA fallback)",
                    cuda_available=False,
                    amp_supported=False,
                )

        if req == "auto":
            if cuda_ok:
                return DeviceInfo(
                    device_string="cuda:0",
                    device_type="cuda",
                    device_name=dev_name,
                    vram_total_mb=vram_total,
                    vram_available_mb=vram_free,
                    cuda_available=True,
                    amp_supported=True,
                )
            else:
                return DeviceInfo(
                    device_string="cpu",
                    device_type="cpu",
                    device_name=dev_name,
                    cuda_available=False,
                    amp_supported=False,
                )

        # Explicit CPU or other
        return DeviceInfo(
            device_string="cpu",
            device_type="cpu",
            device_name="Host CPU",
            cuda_available=cuda_ok,
            amp_supported=False,
        )
