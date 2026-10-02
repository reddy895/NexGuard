"""
NexGuard YOLOv8 Object Detection Subsystem
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from backend.app.config import settings

logger = logging.getLogger("NexGuard.Detector")

# Class color mapping for visually appealing modern CCTV display
CLASS_COLORS = {
    "person": (255, 120, 0),     # Bright blue/orange accent
    "car": (0, 220, 120),        # Vibrant mint green
    "motorcycle": (230, 80, 255), # Purple/Magenta
    "bus": (255, 200, 0),        # Gold / Yellow
    "truck": (0, 180, 255),      # Light Cyan
    "bicycle": (150, 255, 0)     # Neon lime
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
        self.bbox = [float(x) for x in bbox]  # [x1, y1, x2, y2]
        self.track_id = track_id
        
        # Calculate centroid
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
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.YOLO_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing YOLO detector ({self.model_name}) on device: {self.device}")
        
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_name}': {e}")
            self.ready = False
            self.model = None

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Runs YOLO detection on a single frame."""
        if not self.ready or self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            classes=settings.TARGET_CLASSES,
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

        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            class_name = settings.CLASS_NAMES.get(cls_id, f"object_{cls_id}")
            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=xyxy
                )
            )

        return detections

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        show_ids: bool = True
    ) -> np.ndarray:
        """Draws color-coded bounding boxes, class labels, confidence scores, and track IDs."""
        annotated = frame.copy()

        for det in detections:
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)
            x1, y1, x2, y2 = [int(v) for v in det.bbox]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label string
            if det.track_id >= 0 and show_ids:
                label = f"{det.class_name.upper()} #{det.track_id} {det.confidence:.2f}"
            else:
                label = f"{det.class_name.upper()} {det.confidence:.2f}"

            # Text background pill
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fill background pill box
            cv2.rectangle(annotated, (x1, max(0, y1 - h - 6)), (x1 + w + 8, max(h + 6, y1)), color, -1)
            # Text string
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(h + 2, y1 - 4)),
                font,
                font_scale,
                (10, 15, 15),
                thickness,
                lineType=cv2.LINE_AA
            )

        return annotated
