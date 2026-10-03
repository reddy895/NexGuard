"""
NexGuard YOLO Computer Vision Object & Accident Detector
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.config import settings, BASE_DIR, MODELS_DIR

logger = logging.getLogger("NexGuard.Detector")

CLASS_COLORS = {
    "person": (255, 120, 0),     # Bright blue/orange
    "car": (0, 220, 120),        # Mint green
    "motorcycle": (230, 80, 255), # Purple/Magenta
    "bus": (255, 200, 0),        # Gold
    "truck": (0, 180, 255),      # Light Cyan
    "bicycle": (150, 255, 0),    # Neon lime
    "accident": (0, 0, 255)      # Red alert for custom trained accident class
}
DEFAULT_COLOR = (200, 200, 200)


class DetectionResult:
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        bbox: List[float],
        track_id: int = -1
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = float(confidence)
        self.bbox = [float(x) for x in bbox]
        self.track_id = track_id
        self.center = [
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0
        ]
        self.width = max(1.0, self.bbox[2] - self.bbox[0])
        self.height = max(1.0, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(x, 2) for x in self.bbox],
            "track_id": self.track_id,
            "center": [round(x, 2) for x in self.center]
        }


class NexGuardDetector:
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False
        self.model = None
        self.model_name = "Generic YOLOv8"

        self._select_and_load_model(model_path)

    def _select_and_load_model(self, custom_path: str = None):
        if custom_path:
            target = BASE_DIR / custom_path
        else:
            # Prefer custom trained accident model if available
            trained_model = BASE_DIR / settings.TRAINED_MODEL_PATH
            base_model = BASE_DIR / settings.BASE_MODEL_PATH
            root_model = MODELS_DIR / "yolov8n.pt"

            if trained_model.exists():
                target = trained_model
                self.model_name = "Custom Accident YOLO"
            elif base_model.exists():
                target = base_model
                self.model_name = "YOLOv8 Base"
            elif root_model.exists():
                target = root_model
                self.model_name = "YOLOv8 Base"
            else:
                target = root_model
                self.model_name = "YOLOv8 Auto-Download"

        print(f"[+] Loading YOLO model: {target} (Device: {self.device.upper()})")

        try:
            self.model = YOLO(str(target) if target.exists() else "yolov8n.pt")
            self.model.to(self.device)
            self.model_path = target
            self.ready = True
        except Exception as e:
            print(f"NEXGUARD ERROR: Failed to load YOLO model: {e}")
            self.ready = False

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if not self.ready or self.model is None or frame is None:
            return []

        # Predict using imgsz=640
        results = self.model.predict(
            source=frame,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.INFERENCE_IMGSZ,
            verbose=False,
            device=self.device
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names_map = res.names or settings.CLASS_NAMES

        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            class_name = names_map.get(cls_id, f"object_{cls_id}").lower()
            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=xyxy
                )
            )

        return detections
