"""
NexGuard Centralized Configuration System
Provides all parameters for model inference, detection, tracking, temporal accident analysis,
severity scoring, performance tuning, and notification thresholds.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import torch

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "base"
CUSTOM_MODEL_DIR = MODELS_DIR / "custom"
INCIDENTS_DIR = BASE_DIR / "incidents"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "training" / "dataset"

# Ensure essential directories exist
for d in [MODELS_DIR, BASE_MODEL_DIR, CUSTOM_MODEL_DIR, INCIDENTS_DIR, LOGS_DIR, OUTPUTS_DIR, DATASET_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def detect_device() -> str:
    """Detects whether CUDA GPU is available or defaults to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class NexGuardConfig:
    """Centralized NexGuard application configuration."""
    
    # Model Paths
    base_model_path: str = str(BASE_MODEL_DIR / "yolov8n.pt")
    custom_model_path: str = str(CUSTOM_MODEL_DIR / "accident.pt")
    
    # Device & Processing
    device: str = "auto"
    inference_size: int = 640  # Options: 640, 512, 416
    process_every_n_frames: int = 2  # Frame skip (1 = process every frame, 2 = skip every 2nd frame)
    performance_mode: str = "balanced"  # Options: "accuracy", "balanced", "performance"
    
    # Detection Thresholds
    yolo_confidence: float = 0.35
    iou_threshold: float = 0.25
    
    # Tracking Settings
    track_history_length: int = 20
    max_lost_frames: int = 30
    
    # Motion & Collision Thresholds
    proximity_threshold_px: float = 120.0
    collision_iou_threshold: float = 0.15
    sudden_speed_drop_threshold: float = 0.50  # 50% relative speed drop
    direction_change_threshold: float = 45.0   # 45 degrees relative angle shift
    velocity_approach_threshold: float = 15.0  # px/frame
    
    # Temporal Confirmation & Cooldown
    min_collision_frames: int = 3
    accident_candidate_threshold: float = 0.35
    accident_confirmation_frames: int = 5
    accident_cooldown_seconds: int = 30
    
    # Object Classes
    person_classes: Tuple[str, ...] = ("person",)
    vehicle_classes: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck", "bicycle")
    
    # Display Settings
    display_width: int = 1280
    display_height: int = 720
    window_name: str = "NexGuard AI CCTV Accident Detection"
    
    # WhatsApp Configuration
    whatsapp_recipient: str = ""
    alert_cooldown_seconds: int = 30

    def __post_init__(self):
        if self.device == "auto":
            self.device = detect_device()
        self.yolo_confidence = max(0.01, min(1.0, float(self.yolo_confidence)))
        self.iou_threshold = max(0.01, min(1.0, float(self.iou_threshold)))
        self.apply_performance_mode(self.performance_mode)

    def apply_performance_mode(self, mode: str):
        """Applies presets for Accuracy, Balanced, or Performance modes."""
        self.performance_mode = mode.lower()
        if self.performance_mode == "accuracy":
            self.process_every_n_frames = 1
            self.inference_size = 640
        elif self.performance_mode == "performance":
            self.process_every_n_frames = 3
            self.inference_size = 416
        else:  # balanced
            self.process_every_n_frames = 2
            self.inference_size = 640

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model_path": self.base_model_path,
            "custom_model_path": self.custom_model_path,
            "device": self.device,
            "inference_size": self.inference_size,
            "process_every_n_frames": self.process_every_n_frames,
            "performance_mode": self.performance_mode,
            "yolo_confidence": self.yolo_confidence,
            "iou_threshold": self.iou_threshold,
            "track_history_length": self.track_history_length,
            "proximity_threshold_px": self.proximity_threshold_px,
            "collision_iou_threshold": self.collision_iou_threshold,
            "sudden_speed_drop_threshold": self.sudden_speed_drop_threshold,
            "direction_change_threshold": self.direction_change_threshold,
            "min_collision_frames": self.min_collision_frames,
            "accident_candidate_threshold": self.accident_candidate_threshold,
            "accident_confirmation_frames": self.accident_confirmation_frames,
            "accident_cooldown_seconds": self.accident_cooldown_seconds,
            "whatsapp_recipient": self.whatsapp_recipient
        }


# Global configuration singleton instance
config = NexGuardConfig()
