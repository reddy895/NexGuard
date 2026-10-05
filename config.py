"""
NexGuard — Centralized Configuration System
============================================
All configurable parameters are defined here and loaded from environment
variables via python-dotenv. Never hard-code secrets or deployment-specific
values directly in this file.

Usage:
    from config import cfg
    print(cfg.model_path)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence" / "incidents"
LOGS_DIR = BASE_DIR / "logs"
TEST_CLIPS_DIR = BASE_DIR / "test_clips"
DATASETS_DIR = BASE_DIR / "datasets"

# Create runtime directories (not committed, but needed at runtime)
for _d in [EVIDENCE_DIR, LOGS_DIR, TEST_CLIPS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class NexGuardConfig:
    """
    Central configuration for the NexGuard surveillance system.

    All values can be overridden through environment variables
    (see .env.example for full reference).
    """

    # --- Model ---
    model_path: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_MODEL_PATH", "yolov8n.pt")
    )
    device: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_DEVICE", "cpu")
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_CONFIDENCE", "0.40"))
    )
    iou_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_IOU_THRESHOLD", "0.45"))
    )
    inference_size: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_INFERENCE_SIZE", "640"))
    )

    # --- Video Input ---
    source: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_SOURCE", "0")
    )
    max_fps: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_MAX_FPS", "30"))
    )

    # --- Target Classes (COCO names) ---
    vehicle_classes: Tuple[str, ...] = (
        "car", "motorcycle", "bicycle", "bus", "truck"
    )
    person_classes: Tuple[str, ...] = ("person",)

    # --- Tracking ---
    max_lost_frames: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_MAX_LOST_FRAMES", "30"))
    )
    track_history_length: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_TRACK_HISTORY", "30"))
    )

    # --- Accident Detection ---
    accident_window: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_ACCIDENT_WINDOW", "20"))
    )
    accident_threshold: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_ACCIDENT_THRESHOLD", "0.60"))
    )
    accident_confirm_frames: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_ACCIDENT_CONFIRM_FRAMES", "5"))
    )

    # Collision analysis thresholds
    proximity_threshold_px: float = 120.0    # centroid distance for proximity
    collision_iou_threshold: float = 0.10    # box overlap indicating collision
    speed_drop_threshold: float = 0.45       # 45% sudden speed reduction
    direction_change_threshold: float = 45.0  # degrees — sudden direction change
    approach_velocity_threshold: float = 5.0  # px/frame — fast closing speed

    # --- Severity Thresholds (score 0.0–1.0) ---
    severity_low: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_LOW", "0.30"))
    )
    severity_medium: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_MEDIUM", "0.55"))
    )
    severity_high: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_HIGH", "0.75"))
    )
    severity_critical: float = field(
        default_factory=lambda: float(os.getenv("NEXGUARD_SEVERITY_CRITICAL", "0.90"))
    )

    # --- Alerts ---
    alert_cooldown: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_ALERT_COOLDOWN", "60"))
    )

    # --- Evidence ---
    evidence_dir: str = field(
        default_factory=lambda: os.getenv(
            "NEXGUARD_EVIDENCE_DIR",
            str(EVIDENCE_DIR)
        )
    )
    evidence_retention_days: int = field(
        default_factory=lambda: int(os.getenv("NEXGUARD_EVIDENCE_RETENTION_DAYS", "30"))
    )

    # --- Logging ---
    log_level: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_LOG_LEVEL", "INFO").upper()
    )

    # --- WhatsApp (QR login via whatsapp-web.js — no API key needed) ---
    whatsapp_enabled: bool = field(
        default_factory=lambda: os.getenv("NEXGUARD_WHATSAPP_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    whatsapp_recipients: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("NEXGUARD_WHATSAPP_RECIPIENTS", "9591152862").split(",")
            if r.strip()
        ]
    )
    whatsapp_api_key: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_WHATSAPP_API_KEY", "")
    )

    # Alert routing by severity
    alert_low_contacts: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("NEXGUARD_ALERT_LOW_CONTACTS", "").split(",")
            if r.strip()
        ]
    )
    alert_medium_contacts: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("NEXGUARD_ALERT_MEDIUM_CONTACTS", "").split(",")
            if r.strip()
        ]
    )
    alert_high_contacts: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("NEXGUARD_ALERT_HIGH_CONTACTS", "").split(",")
            if r.strip()
        ]
    )
    alert_critical_contacts: List[str] = field(
        default_factory=lambda: [
            r.strip()
            for r in os.getenv("NEXGUARD_ALERT_CRITICAL_CONTACTS", "").split(",")
            if r.strip()
        ]
    )

    # --- Display ---
    display_width: int = 1280
    display_height: int = 720
    window_name: str = "NexGuard — AI Safety Monitoring"

    # --- Location (metadata for alerts) ---
    location_name: str = field(
        default_factory=lambda: os.getenv("NEXGUARD_LOCATION_NAME", "Unknown Location")
    )

    def get_contacts_for_severity(self, severity: str) -> List[str]:
        """
        Returns the appropriate contact list based on severity level.
        Falls back to all recipients if specific list is empty.
        """
        mapping = {
            "LOW": self.alert_low_contacts,
            "MEDIUM": self.alert_medium_contacts,
            "HIGH": self.alert_high_contacts,
            "CRITICAL": self.alert_critical_contacts,
        }
        contacts = mapping.get(severity.upper(), [])
        if not contacts:
            contacts = self.whatsapp_recipients
        return contacts

    def validate(self) -> None:
        """
        Validates critical configuration values and raises clear errors.
        Call this at startup to catch misconfiguration early.
        """
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(
                f"NEXGUARD_CONFIDENCE must be between 0.0 and 1.0, got: {self.confidence_threshold}"
            )
        if self.iou_threshold < 0.0 or self.iou_threshold > 1.0:
            raise ValueError(
                f"NEXGUARD_IOU_THRESHOLD must be between 0.0 and 1.0, got: {self.iou_threshold}"
            )
        if self.accident_threshold < 0.0 or self.accident_threshold > 1.0:
            raise ValueError(
                f"NEXGUARD_ACCIDENT_THRESHOLD must be 0.0–1.0, got: {self.accident_threshold}"
            )
        if self.whatsapp_enabled and not self.whatsapp_recipients:
            raise ValueError(
                "NEXGUARD_WHATSAPP_ENABLED=true but NEXGUARD_WHATSAPP_RECIPIENTS is empty. "
                "Set at least one recipient phone number."
            )

    @property
    def all_target_classes(self) -> Tuple[str, ...]:
        """Combined set of all object classes NexGuard is interested in."""
        return self.vehicle_classes + self.person_classes


# ---------------------------------------------------------------------------
# Singleton configuration instance
# ---------------------------------------------------------------------------
cfg = NexGuardConfig()
