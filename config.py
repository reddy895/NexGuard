"""
NexGuard — Centralized Configuration
======================================
All configurable parameters live here and are loaded from environment
variables via python-dotenv.  Never hard-code secrets in this file.

Usage:
    from config import cfg
    print(cfg.model_path)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "models"
SAMPLE_VIDEOS_DIR: Path = BASE_DIR / "sample_videos"
LOGS_DIR: Path = BASE_DIR / "logs"
DATASETS_DIR: Path = BASE_DIR / "datasets"

for _d in [LOGS_DIR, MODELS_DIR, SAMPLE_VIDEOS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class NexGuardConfig:
    """
    Central configuration for the NexGuard local detection system.

    All values can be overridden through environment variables
    (see .env.example for full reference).
    """

    # ── Model ─────────────────────────────────────────────────────────────
    model_path: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_MODEL_PATH", "yolov8n.pt")
    )
    device: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_DEVICE", "auto")
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_CONFIDENCE", "0.30"))
    )
    iou_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_IOU_THRESHOLD", "0.45"))
    )
    inference_size: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_INFERENCE_SIZE", "640"))
    )

    # ── Target classes ────────────────────────────────────────────────────
    vehicle_classes: Tuple[str, ...] = (
        "car", "motorcycle", "bicycle", "bus", "truck"
    )
    person_classes: Tuple[str, ...] = ("person",)
    accident_classes: Tuple[str, ...] = (
        "accident", "crash", "collision", "severe_accident",
        "moderate_accident", "near_miss", "overturned_vehicle",
    )

    # ── Tracking ──────────────────────────────────────────────────────────
    max_lost_frames: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_MAX_LOST_FRAMES", "30"))
    )
    track_history_length: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_TRACK_HISTORY", "30"))
    )

    # ── Accident detection ────────────────────────────────────────────────
    accident_window: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_ACCIDENT_WINDOW", "20"))
    )
    accident_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_ACCIDENT_THRESHOLD", "0.55"))
    )
    accident_confirm_frames: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_ACCIDENT_CONFIRM_FRAMES", "4"))
    )
    accident_cooldown_frames: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_ACCIDENT_COOLDOWN_FRAMES", "60"))
    )

    # Collision geometry thresholds
    proximity_threshold_px: float = 130.0
    collision_iou_threshold: float = 0.08
    speed_drop_threshold: float = 0.45
    direction_change_threshold: float = 45.0
    approach_velocity_threshold: float = 4.0

    # ── Severity thresholds (score 0.0–1.0) ──────────────────────────────
    severity_low: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_LOW", "0.25"))
    )
    severity_medium: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_MEDIUM", "0.50"))
    )
    severity_high: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_HIGH", "0.72"))
    )

    # ── Display ───────────────────────────────────────────────────────────
    window_name: str = "NexGuard — AI Road Accident Detection"
    display_fps_cap: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_DISPLAY_FPS", "30"))
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_LOG_LEVEL", "WARNING").upper()
    )

    def resolve_device(self) -> str:
        """
        Resolves 'auto' to 'cuda' if available, otherwise 'cpu'.
        Always returns a concrete device string.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def validate(self) -> None:
        """Raises ValueError on misconfigured values."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"NEXGUARD_CONFIDENCE must be 0.0–1.0, got {self.confidence_threshold}"
            )
        if not (0.0 <= self.iou_threshold <= 1.0):
            raise ValueError(
                f"NEXGUARD_IOU_THRESHOLD must be 0.0–1.0, got {self.iou_threshold}"
            )
        if not (0.0 <= self.accident_threshold <= 1.0):
            raise ValueError(
                f"NEXGUARD_ACCIDENT_THRESHOLD must be 0.0–1.0, got {self.accident_threshold}"
            )

    @property
    def all_target_classes(self) -> Tuple[str, ...]:
        """Combined set of all classes the pipeline monitors."""
        return self.vehicle_classes + self.person_classes + self.accident_classes


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
cfg = NexGuardConfig()
