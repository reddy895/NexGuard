"""
NexGuard YOLO Detector Module
Handles object detection model loading, hardware acceleration, inference, confidence filtering,
and dynamic class map resolution.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.logger import get_logger
from src.utils import detect_device, get_device_info

logger = get_logger()


class NexGuardDetector:
    """
    Object detection engine encapsulating Ultralytics YOLO model inference.
    Supports auto device selection, threshold filtering, and dynamic class metadata.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Device selection
        if device == "auto":
            self.device = detect_device()
        else:
            self.device = device.lower()

        self.dev_type, self.dev_name = get_device_info()
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.is_loaded = False

        self._load_model()

    def _load_model(self) -> None:
        """Loads the Ultralytics YOLO model engine."""
        logger.info(f"Loading YOLO detection model '{self.model_path}' on device '{self.device}'...")

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            
            # Transfer model to device if specified
            if hasattr(self.model, "to"):
                try:
                    self.model.to(self.device)
                except Exception as dev_err:
                    logger.warning(f"Could not move model to {self.device}: {dev_err}. Falling back to default.")

            # Load dynamic class names mapping
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
            else:
                # Default COCO sample map fallback
                self.class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

            self.is_loaded = True
            logger.info(f"[✓] YOLO Model successfully loaded with {len(self.class_names)} target classes.")

        except ImportError:
            logger.error("[ERROR] Ultralytics module not installed. Running detector in fallback mode.")
            self._setup_fallback()
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model '{self.model_path}': {e}. Using fallback engine.")
            self._setup_fallback()

    def _setup_fallback(self) -> None:
        """Configures a lightweight fallback engine if PyTorch/Ultralytics is unavailable."""
        self.model = None
        self.class_names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.is_loaded = False

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection inference on an RGB/BGR image frame.

        Returns list of detection dictionaries:
        [
            {
                "box": [x1, y1, x2, y2],
                "confidence": 0.92,
                "class_id": 2,
                "class_name": "car"
            }, ...
        ]
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        if not self.is_loaded or self.model is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            detections: List[Dict[str, Any]] = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                    detections.append({
                        "box": [round(c, 1) for c in xyxy],
                        "confidence": round(conf, 4),
                        "class_id": cls_id,
                        "class_name": cls_name
                    })

            return detections

        except Exception as err:
            logger.error(f"Inference error: {err}")
            return []

    def get_metadata(self) -> Dict[str, Any]:
        """Exposes detector configuration and model metadata."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "device_name": self.dev_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "class_count": len(self.class_names),
            "is_loaded": self.is_loaded
        }
