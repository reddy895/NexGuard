"""
NexGuard — AI CCTV Accident Detection Configuration
Centralized configuration parameters for YOLO detection, multi-object tracking,
max 15 FPS frame rate limiting, motion gesture classifier, and WhatsApp messaging.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Root Paths
BASE_DIR = Path(__file__).resolve().parent
TEST_CLIPS_DIR = BASE_DIR / "test_clips"
WHATSAPP_BOT_DIR = BASE_DIR / "whatsapp_bot"
MODEL_PATH = BASE_DIR / "yolov8n.pt"

# Ensure directories exist
TEST_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
WHATSAPP_BOT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # Model & Frame Rate Controls
    yolo_model_path: str = str(MODEL_PATH)
    device: str = "cpu"  # 'cuda' or 'cpu'
    max_fps: int = 15  # MAX 15 FPS restriction strictly enforced
    yolo_confidence: float = 0.35
    iou_threshold: float = 0.30
    inference_size: int = 640

    # Motion & Collision Dynamics Thresholds
    proximity_threshold_px: float = 100.0
    collision_iou_threshold: float = 0.15
    speed_drop_threshold: float = 0.45       # 45% sudden drop in speed
    direction_change_threshold: float = 40.0 # 40 degrees angular change
    accident_confirm_frames: int = 4         # Temporal confirmation over 4 consecutive frames

    # Tracker Settings
    max_lost_frames: int = 25
    track_history_length: int = 15

    # Gesture / Motion Classifier Paths
    gesture_model_joblib: str = str(BASE_DIR / "gesture_classifier.joblib")
    gesture_model_npz: str = str(BASE_DIR / "gesture_classifier_fast.npz")

    # Display & Visual UI
    display_width: int = 1280
    display_height: int = 720
    window_name: str = "NexGuard AI CCTV Accident System (Max 15 FPS)"

    # WhatsApp Alerting Settings
    whatsapp_recipient: str = os.getenv("WHATSAPP_RECIPIENT", "")
    whatsapp_server_port: int = 3001
    alert_cooldown_seconds: int = 20

    # Target Object Classes (COCO IDs / Names)
    vehicle_classes: Tuple[str, ...] = ("car", "motorcycle", "bus", "truck", "bicycle")
    person_classes: Tuple[str, ...] = ("person",)


config = Config()
