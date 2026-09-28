"""
NexGuard Configuration System
Provides configuration parameters for model inference, input sources, display, and thresholds.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.utils import detect_device


@dataclass
class NexGuardConfig:
    """NexGuard application configuration parameters."""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    camera_index: int = 0
    input_size: int = 640
    frame_skip: int = 0
    output_dir: str = "outputs"
    log_dir: str = "logs"
    window_name: str = "NexGuard Live Edge AI Surveillance"
    refresh_rate_sec: float = 1.0

    def __post_init__(self):
        """Resolve device auto-detection and directory paths."""
        if self.device == "auto":
            self.device = detect_device()
        self.confidence_threshold = max(0.0, min(1.0, float(self.confidence_threshold)))
        self.iou_threshold = max(0.0, min(1.0, float(self.iou_threshold)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "camera_index": self.camera_index,
            "input_size": self.input_size,
            "frame_skip": self.frame_skip,
            "output_dir": self.output_dir,
            "log_dir": self.log_dir,
            "window_name": self.window_name,
            "refresh_rate_sec": self.refresh_rate_sec,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NexGuardConfig":
        """Instantiate config from dictionary."""
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


# Default global instance
DEFAULT_CONFIG = NexGuardConfig()
