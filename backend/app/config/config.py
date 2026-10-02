"""
NexGuard — System Configuration & Environment Loader
"""

import os
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INCIDENTS_DIR = DATA_DIR / "incidents"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class NexGuardSettings(BaseModel):
    # System
    PROJECT_NAME: str = "NexGuard"
    VERSION: str = "2.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # YOLO & Detection
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    TARGET_CLASSES: List[int] = [0, 1, 2, 3, 5, 7]  # COCO: person, bicycle, car, motorcycle, bus, truck
    CLASS_NAMES: Dict[int, str] = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    # Accident Detection Rule Engine Parameters
    PROXIMITY_THRESHOLD_PX: float = float(os.getenv("PROXIMITY_THRESHOLD_PX", "60.0"))
    VELOCITY_APPROACH_THRESHOLD: float = float(os.getenv("VELOCITY_APPROACH_THRESHOLD", "8.0"))
    MIN_TEMPORAL_FRAMES: int = int(os.getenv("MIN_TEMPORAL_FRAMES", "3"))
    POST_COLLISION_STOP_FRAMES: int = int(os.getenv("POST_COLLISION_STOP_FRAMES", "5"))
    ACCIDENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", "0.60"))
    
    # Severity Thresholds (Score ranges)
    SEVERITY_THRESHOLDS: Dict[str, float] = {
        "CRITICAL": 0.85,
        "HIGH": 0.70,
        "MEDIUM": 0.50,
        "LOW": 0.30
    }
    
    # WhatsApp Alert System
    WHATSAPP_API_TOKEN: str = os.getenv("WHATSAPP_API_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WHATSAPP_RECIPIENT_NUMBER: str = os.getenv("WHATSAPP_RECIPIENT_NUMBER", "")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))

    @property
    def whatsapp_configured(self) -> bool:
        return bool(
            self.WHATSAPP_API_TOKEN and
            self.WHATSAPP_PHONE_NUMBER_ID and
            self.WHATSAPP_RECIPIENT_NUMBER
        )


settings = NexGuardSettings()
