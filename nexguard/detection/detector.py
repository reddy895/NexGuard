"""
NexGuard — YOLO Object Detector
=================================
Wraps Ultralytics YOLO for person and vehicle detection.

Design decisions:
- Single model instance, no reload across frames.
- Filters detections to only NexGuard target classes (persons, vehicles).
- Returns structured Detection objects rather than raw YOLO tensors.
- Gracefully handles CUDA unavailability by falling back to CPU.
- verbose=False to suppress per-frame YOLO logging noise.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from config import cfg, MAX_FPS
from nexguard.models.schemas import Detection, ModelDiagnosticInfo, ModelType
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.detection")


class ObjectDetector:
    """
    YOLO-based object detector for NexGuard surveillance.

    Attributes:
        model_path:  Path to the YOLO weights file.
        device:      Inference device ('cuda', 'cpu', 'mps').
        conf:        Confidence threshold [0, 1].
        iou:         NMS IoU threshold [0, 1].
        imgsz:       Input resolution fed to the model.
        model:       Loaded YOLO model instance.
        model_ready: True if the model loaded successfully.
        diagnostic_info: ModelDiagnosticInfo containing detailed inspection metadata.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        self.model_path = model_path or cfg.model_path
        self.device = device or cfg.device
        self.conf = conf if conf is not None else cfg.confidence_threshold
        self.iou = iou if iou is not None else cfg.iou_threshold
        self.imgsz = imgsz or cfg.inference_size

        self.model = None
        self.model_ready = False
        self._class_names: dict = {}
        self.diagnostic_info: Optional[ModelDiagnosticInfo] = None

        self._load_model()

    def _load_model(self) -> None:
        """Loads the YOLO model. Falls back to CPU if CUDA is unavailable."""
        dataset_p = Path(cfg.accident_dataset_path)
        dataset_status = "FOUND" if dataset_p.exists() else "MISSING (datasets/accident/data.yaml not found)"

        try:
            from ultralytics import YOLO
        except ImportError:
            err_msg = "ultralytics package not found. Install with: pip install ultralytics"
            log.error(err_msg)
            self.diagnostic_info = ModelDiagnosticInfo(
                model_path=self.model_path,
                device=self.device,
                model_type=ModelType.UNKNOWN,
                class_count=0,
                classes={},
                accident_classes_found=[],
                inference_size=self.imgsz,
                confidence_threshold=self.conf,
                max_fps=cfg.max_fps,
                dataset_status=dataset_status,
                dataset_path=str(dataset_p),
                ready=False,
                error_message=err_msg,
            )
            return

        # Resolve model path — download if not local
        model_p = Path(self.model_path)
        if not model_p.exists():
            log.info(
                f"Model not found at {self.model_path} — "
                "Ultralytics will auto-download the weights."
            )

        # Attempt CUDA, fall back to CPU
        device = self.device
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                log.warning("CUDA requested but unavailable — falling back to CPU.")
                device = "cpu"
                self.device = "cpu"
        except ImportError:
            log.warning("PyTorch not available — using CPU inference.")
            device = "cpu"
            self.device = "cpu"

        try:
            self.model = YOLO(self.model_path)
            # Warm-up inference on a blank frame
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device=device)
            self._class_names = self.model.names or {}
            self.model_ready = True

            # Inspect model classes for accident detection capability
            found_accident = [
                name for name in self._class_names.values()
                if any(acc.lower() in name.lower() for acc in cfg.accident_classes)
            ]
            found_vehicles = [
                name for name in self._class_names.values()
                if name in cfg.vehicle_classes
            ]

            if found_accident and found_vehicles:
                model_type = ModelType.HYBRID_DETECTOR
            elif found_accident:
                model_type = ModelType.ACCIDENT_DETECTOR
            else:
                model_type = ModelType.GENERIC_DETECTOR

            self.diagnostic_info = ModelDiagnosticInfo(
                model_path=str(model_p.resolve() if model_p.exists() else model_p),
                device=device,
                model_type=model_type,
                class_count=len(self._class_names),
                classes=dict(self._class_names),
                accident_classes_found=found_accident,
                inference_size=self.imgsz,
                confidence_threshold=self.conf,
                max_fps=cfg.max_fps,
                dataset_status=dataset_status,
                dataset_path=str(dataset_p),
                ready=True,
            )

            if cfg.strict_accident_model and model_type == ModelType.GENERIC_DETECTOR:
                err_msg = (
                    "NEXGUARD ERROR: ACCIDENT DATASET REQUIRED — "
                    f"Model '{self.model_path}' is a generic object detector with {len(self._class_names)} classes "
                    "and contains no accident-specific classes. Fine-tune or supply an accident model."
                )
                log.error(err_msg)
                self.model_ready = False
                self.diagnostic_info.ready = False
                self.diagnostic_info.error_message = err_msg
                return

            log.info(
                f"Detector ready — model={self.model_path} ({model_type.value})  "
                f"device={device}  conf={self.conf:.2f}  iou={self.iou:.2f}  imgsz={self.imgsz}"
            )
        except Exception as e:
            err_msg = f"Failed to load YOLO model: {e}"
            log.error(err_msg)
            self.model_ready = False
            self.diagnostic_info = ModelDiagnosticInfo(
                model_path=self.model_path,
                device=device,
                model_type=ModelType.UNKNOWN,
                class_count=0,
                classes={},
                accident_classes_found=[],
                inference_size=self.imgsz,
                confidence_threshold=self.conf,
                max_fps=cfg.max_fps,
                dataset_status=dataset_status,
                dataset_path=str(dataset_p),
                ready=False,
                error_message=err_msg,
            )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on a single frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            List of Detection objects filtered to NexGuard target classes.
        """
        if not self.model_ready or self.model is None:
            return []

        try:
            results = self.model(
                frame,
                verbose=False,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
            )[0]
        except Exception as e:
            log.warning(f"Detection error on frame: {e}")
            return []

        detections: List[Detection] = []
        target_classes = set(cfg.all_target_classes)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

            is_acc = (
                cls_name in cfg.accident_classes
                or any(acc.lower() in cls_name.lower() for acc in cfg.accident_classes)
            )

            # Accept if in target classes, or matches accident class, or model is dedicated accident detector
            if (
                cls_name not in target_classes
                and not is_acc
                and self.diagnostic_info
                and self.diagnostic_info.model_type != ModelType.ACCIDENT_DETECTOR
            ):
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            detections.append(
                Detection(
                    bbox=(
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3]),
                    ),
                    class_name=cls_name,
                    confidence=conf,
                    class_id=cls_id,
                    is_accident=is_acc,
                )
            )

        return detections

    @property
    def class_names(self) -> dict:
        """Returns the model's class name mapping."""
        return self._class_names

    @property
    def is_accident_model(self) -> bool:
        """Returns True if the model has native accident detection classes."""
        if not self.diagnostic_info:
            return False
        return self.diagnostic_info.model_type in (
            ModelType.ACCIDENT_DETECTOR,
            ModelType.HYBRID_DETECTOR,
        )

    def get_diagnostic_info(self) -> ModelDiagnosticInfo:
        """Returns startup diagnostics information."""
        if self.diagnostic_info is not None:
            return self.diagnostic_info
        return ModelDiagnosticInfo(
            model_path=self.model_path,
            device=self.device,
            model_type=ModelType.UNKNOWN,
            class_count=len(self._class_names),
            classes=dict(self._class_names),
            accident_classes_found=[],
            inference_size=self.imgsz,
            confidence_threshold=self.conf,
            max_fps=cfg.max_fps,
            dataset_status="UNKNOWN",
            ready=self.model_ready,
        )

    def __repr__(self) -> str:
        status = "READY" if self.model_ready else "FAILED"
        return (
            f"ObjectDetector(model={self.model_path!r}, device={self.device!r}, "
            f"conf={self.conf}, status={status})"
        )
