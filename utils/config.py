"""
NexGuard — Centralized Configuration & Environment Loader
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INCIDENTS_DIR = DATA_DIR / "incidents"
MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
BASE_MODELS_DIR = MODELS_DIR / "base"

# Ensure required directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)


class NexGuardConfig:
    PROJECT_NAME: str = "NexGuard"
    VERSION: str = "2.1.0"

    # YOLO Model Paths
    BASE_MODEL_PATH: str = "models/base/yolov8n.pt"
    TRAINED_MODEL_PATH: str = "models/trained/accident_best.pt"
    DEFAULT_YOLO_MODEL: str = os.getenv("YOLO_MODEL_PATH", "models/base/yolov8n.pt")

    # Inference & Window Dimensions
    INFERENCE_IMGSZ: int = 640
    DISPLAY_WIDTH: int = 1280
    DISPLAY_HEIGHT: int = 720

    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # COCO target classes + custom accident class
    TARGET_CLASSES = [0, 1, 2, 3, 5, 7]
    CLASS_NAMES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    # Accident Rule Engine Parameters
    PROXIMITY_THRESHOLD_PX: float = float(os.getenv("PROXIMITY_THRESHOLD_PX", "65.0"))
    VELOCITY_APPROACH_THRESHOLD: float = float(os.getenv("VELOCITY_APPROACH_THRESHOLD", "8.0"))
    MIN_TEMPORAL_FRAMES: int = int(os.getenv("MIN_TEMPORAL_FRAMES", "3"))
    POST_COLLISION_STOP_FRAMES: int = int(os.getenv("POST_COLLISION_STOP_FRAMES", "5"))
    ACCIDENT_CONFIDENCE_THRESHOLD: float = float(os.getenv("ACCIDENT_CONFIDENCE_THRESHOLD", "0.55"))

    # WhatsApp Configuration
    WHATSAPP_RECIPIENT: str = os.getenv("WHATSAPP_RECIPIENT", "")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))


settings = NexGuardConfig()
