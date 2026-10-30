"""
NexGuard — Model Selector
=========================
Selects optimal YOLO model architecture and size based on:
- Detected hardware (GPU memory availability or CPU execution)
- Task type (classification vs detection)
- Latency and throughput requirements for real-time CCTV analysis
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelSelectionResult:
    model_name: str
    architecture: str
    task: str
    recommended_batch_size: int
    recommended_imgsz: int
    device: str
    rationale: str

class ModelSelector:
    """
    Selects suitable model architecture based on available system hardware.
    """

    @staticmethod
    def get_available_vram_gb() -> float:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    @classmethod
    def select_model(
        cls,
        task: str = "classify",
        force_device: Optional[str] = None,
    ) -> ModelSelectionResult:
        vram = cls.get_available_vram_gb()
        device = force_device if force_device else ("cuda" if vram > 0 else "cpu")

        if task == "classify":
            if vram > 8.0:
                return ModelSelectionResult(
                    model_name="yolov8s-cls.pt",
                    architecture="YOLOv8-Small-Classifier",
                    task="classify",
                    recommended_batch_size=32,
                    recommended_imgsz=224,
                    device=device,
                    rationale=f"High GPU memory detected ({vram:.1f} GB VRAM). Using Small classifier backbone.",
                )
            else:
                return ModelSelectionResult(
                    model_name="yolov8n-cls.pt",
                    architecture="YOLOv8-Nano-Classifier",
                    task="classify",
                    recommended_batch_size=16 if device == "cuda" else 8,
                    recommended_imgsz=224,
                    device=device,
                    rationale=f"Edge/limited VRAM or CPU environment ({vram:.1f} GB VRAM). Using Nano classifier for low latency.",
                )
        else:
            if vram > 8.0:
                return ModelSelectionResult(
                    model_name="yolov8s.pt",
                    architecture="YOLOv8-Small-Detector",
                    task="detect",
                    recommended_batch_size=16,
                    recommended_imgsz=640,
                    device=device,
                    rationale=f"High GPU memory ({vram:.1f} GB VRAM). Using Small object detector.",
                )
            else:
                return ModelSelectionResult(
                    model_name="yolov8n.pt",
                    architecture="YOLOv8-Nano-Detector",
                    task="detect",
                    recommended_batch_size=8 if device == "cuda" else 4,
                    recommended_imgsz=640,
                    device=device,
                    rationale=f"Resource-constrained environment ({vram:.1f} GB VRAM). Using Nano object detector.",
                )
