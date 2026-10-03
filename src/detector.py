"""
NexGuard YOLO Object Detection Module
Provides unified YOLOv8 inference wrapper with configurable confidence threshold,
CUDA/CPU hardware acceleration detection, and optional custom accident model integration.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from config import config
from utils.logger import logger
from utils.geometry import get_bbox_center


class DetectionObject:
    """Represents a single object detected by YOLO."""

    def __init__(self, bbox: List[float], confidence: float, class_id: int, class_name: str):
        self.bbox: List[float] = [float(b) for b in bbox]
        self.confidence: float = float(confidence)
        self.class_id: int = int(class_id)
        self.class_name: str = str(class_name).lower()
        self.center: Tuple[float, float] = get_bbox_center(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": self.center
        }


class YOLOObjectDetector:
    """YOLOv8 Detection Engine supporting Base and Optional Custom Accident Models."""

    def __init__(
        self,
        base_model_path: Optional[str] = None,
        custom_model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None
    ):
        self.conf_threshold = confidence_threshold or config.yolo_confidence
        self.device = device or config.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve Base Model Path
        self.base_model_path = Path(base_model_path or config.base_model_path)
        if not self.base_model_path.exists():
            # Fallback to local root model or download default yolov8n.pt
            alt_base = Path("yolov8n.pt")
            if alt_base.exists():
                self.base_model_path = alt_base

        self.model_name = self.base_model_path.name
        self.base_model: Optional[YOLO] = None
        self.custom_model: Optional[YOLO] = None
        self.custom_model_loaded = False
        self.ready = False

        self._load_models(custom_model_path or config.custom_model_path)

    def _load_models(self, custom_model_path: str):
        """Loads base YOLO model and optionally custom accident model."""
        try:
            model_target = str(self.base_model_path) if self.base_model_path.exists() else "yolov8n.pt"
            logger.info(f"Loading Base YOLO model: {model_target} (Device: {self.device})")
            self.base_model = YOLO(model_target)
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to load Base YOLO model: {e}")
            self.ready = False

        # Load Custom Accident Model if present
        custom_path = Path(custom_model_path)
        if custom_path.exists() and custom_path.is_file():
            try:
                logger.info(f"Loading Custom Accident YOLO model: {custom_path}")
                self.custom_model = YOLO(str(custom_path))
                self.custom_model_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load custom accident model at {custom_path}: {e}")
                self.custom_model_loaded = False

    def detect(self, frame: np.ndarray, imgsz: Optional[int] = None) -> List[DetectionObject]:
        """Runs YOLO object detection on frame and returns filtered DetectionObjects."""
        if not self.ready or self.base_model is None or frame is None:
            return []

        img_size = imgsz or config.inference_size
        results = self.base_model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            imgsz=img_size,
            verbose=False
        )

        detections: List[DetectionObject] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        names = res.names
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = names.get(cls_id, f"cls_{cls_id}")

            # Include target vehicles and persons
            target_classes = set(config.person_classes) | set(config.vehicle_classes) | {"accident"}
            if cls_name.lower() in target_classes:
                detections.append(DetectionObject(
                    bbox=xyxy,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                ))

        # Run custom accident model inference if available
        if self.custom_model_loaded and self.custom_model is not None:
            try:
                custom_results = self.custom_model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    device=self.device,
                    imgsz=img_size,
                    verbose=False
                )
                if custom_results and custom_results[0].boxes is not None:
                    c_res = custom_results[0]
                    c_names = c_res.names
                    for box in c_res.boxes:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = c_names.get(cls_id, "accident")
                        detections.append(DetectionObject(
                            bbox=xyxy,
                            confidence=conf,
                            class_id=cls_id,
                            class_name="accident"
                        ))
            except Exception as e:
                logger.warning(f"Custom model inference error: {e}")

        return detections
