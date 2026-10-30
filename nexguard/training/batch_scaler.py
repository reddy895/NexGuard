"""
NexGuard — Memory-Aware Batch Sizing
===================================
Dynamically scales mini-batch sizes according to available system memory
(VRAM on GPU, RAM on CPU) and image resolution to prevent Out Of Memory (OOM) faults.
"""

from __future__ import annotations

import os
from typing import Tuple
from nexguard.training.device import DeviceManager
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.batch_scaler")

class MemoryAwareBatchScaler:
    """
    Computes safe, conservative batch sizes matching available compute resources.
    """

    @classmethod
    def get_system_ram_gb(cls) -> float:
        try:
            # Read MemTotal from /proc/meminfo on Linux
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except Exception:
            pass
        return 8.0  # Default fallback assumption

    @classmethod
    def calculate_batch_size(
        cls,
        image_size: int = 224,
        task: str = "classify",
        device_override: str = "auto",
        user_requested_batch: int = 0,
    ) -> int:
        if user_requested_batch > 0:
            return user_requested_batch

        dev_info = DeviceManager.resolve_device(device_override)

        if dev_info.cuda_available:
            vram_mb = dev_info.vram_available_mb
            # For 224x224 classification on GPU
            if task == "classify":
                if vram_mb >= 8000:
                    batch = 64
                elif vram_mb >= 3500:
                    batch = 32
                elif vram_mb >= 1800:
                    batch = 16
                else:
                    batch = 8
            else:  # 640x640 detection
                if vram_mb >= 8000:
                    batch = 16
                elif vram_mb >= 3500:
                    batch = 8
                else:
                    batch = 4
        else:
            ram_gb = cls.get_system_ram_gb()
            if task == "classify":
                batch = 16 if ram_gb >= 8.0 else 8
            else:
                batch = 4 if ram_gb >= 8.0 else 2

        log.info(
            "Batch sizing: task=%s imgsz=%d dev=%s -> batch_size=%d",
            task, image_size, dev_info.device_type, batch,
        )
        return batch
