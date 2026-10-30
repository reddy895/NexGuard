"""
NexGuard — YOLO Object Detector
=================================
Wraps Ultralytics YOLO for person and vehicle detection.

Design notes:
- Single model instance, no reload across frames.
- Filters detections to NexGuard target classes only.
- Returns structured Detection objects, not raw YOLO tensors.
- GPU auto-selection with CPU fallback.
- verbose=False suppresses per-frame YOLO log noise.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import cfg
from nexguard.models.schemas import Detection, ModelDiagnosticInfo, ModelType
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.detector")


class ObjectDetector:
    """
    YOLO-based object detector for NexGuard road-accident detection.

    Attributes:
        model_path:     Path to YOLO weights.
        device:         Inference device string ('cuda', 'cpu', 'mps').
        conf:           Detection confidence threshold [0, 1].
        iou:            NMS IoU threshold [0, 1].
        imgsz:          Input resolution fed to the model.
        model_ready:    True when the model has loaded successfully.
        last_inference_time_ms: Milliseconds spent on the last inference call.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        classifier_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or cfg.model_path
        self.device = device or cfg.resolve_device()
        self.conf = conf if conf is not None else cfg.confidence_threshold
        self.iou = iou if iou is not None else cfg.iou_threshold
        self.imgsz = imgsz or cfg.inference_size
        self.classifier_path = classifier_path or str(Path("models/object_detection/best.pt"))

        self.model = None
        self.classifier_model = None
        self.model_ready: bool = False
        self._class_names: dict = {}
        self.diagnostic_info: Optional[ModelDiagnosticInfo] = None
        self.last_inference_time_ms: float = 0.0
        self._lock = threading.Lock()

        self._load_model()
        self._load_classifier()

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Loads the YOLO model. Falls back to CPU if CUDA is unavailable."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._set_failed(
                "ultralytics package not found. "
                "Install with: pip install ultralytics"
            )
            return

        model_p = Path(self.model_path)
        if not model_p.exists():
            log.info(
                "Model weights not found at '%s' — "
                "Ultralytics will attempt to download them.",
                self.model_path,
            )

        # CUDA availability guard
        device = self.device
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                log.warning("CUDA requested but unavailable — falling back to CPU.")
                device = "cpu"
                self.device = "cpu"
        except ImportError:
            device = "cpu"
            self.device = "cpu"

        try:
            self.model = YOLO(self.model_path)
            # Warm-up pass to pre-compile kernels
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device=device)
            self._class_names = self.model.names or {}
            self.model_ready = True

            # Classify model by class names
            found_acc = [
                n for n in self._class_names.values()
                if any(a.lower() in n.lower() for a in cfg.accident_classes)
            ]
            found_veh = [
                n for n in self._class_names.values()
                if n in cfg.vehicle_classes
            ]
            if found_acc and found_veh:
                mtype = ModelType.HYBRID_DETECTOR
            elif found_acc:
                mtype = ModelType.ACCIDENT_DETECTOR
            else:
                mtype = ModelType.GENERIC_DETECTOR

            self.diagnostic_info = ModelDiagnosticInfo(
                model_path=str(model_p.resolve() if model_p.exists() else model_p),
                device=device,
                model_type=mtype,
                class_count=len(self._class_names),
                classes=dict(self._class_names),
                accident_classes_found=found_acc,
                inference_size=self.imgsz,
                confidence_threshold=self.conf,
                dataset_status="N/A" if not found_acc else "embedded",
                ready=True,
            )
            log.info(
                "Detector ready — model=%s (%s) device=%s conf=%.2f",
                self.model_path, mtype.value, device, self.conf,
            )
        except Exception as exc:
            self._set_failed(f"Failed to load YOLO model: {exc}")

    def _set_failed(self, message: str) -> None:
        """Sets the detector into a failed state with a diagnostic message."""
        log.error(message)
        self.model_ready = False
        self.diagnostic_info = ModelDiagnosticInfo(
            model_path=self.model_path,
            device=self.device,
            model_type=ModelType.UNKNOWN,
            class_count=0,
            classes={},
            accident_classes_found=[],
            inference_size=self.imgsz,
            confidence_threshold=self.conf,
            dataset_status="UNAVAILABLE",
            ready=False,
            error_message=message,
        )

    def _load_classifier(self) -> None:
        """Loads optional trained accident classifier weights if present."""
        cp = Path(self.classifier_path)
        if not cp.exists() and Path("models/accident_classifier.pt").exists():
            cp = Path("models/accident_classifier.pt")

        if cp.exists():
            try:
                from ultralytics import YOLO
                self.classifier_model = YOLO(str(cp))
                log.info("Accident classifier loaded successfully from: %s", cp)
            except Exception as exc:
                log.warning("Could not load accident classifier from %s: %s", cp, exc)
                self.classifier_model = None

    def classify_accident(self, frame: np.ndarray) -> Optional[Tuple[float, str]]:
        """
        Runs frame classification using trained accident model weights.
        Returns: (confidence, class_name) or None if classifier is not loaded.
        """
        if self.classifier_model is None:
            return None
        processed = self._preprocess(frame)
        if processed is None:
            return None
        try:
            res = self.classifier_model(processed, verbose=False, device=self.device, imgsz=224)[0]
            top1_idx = int(res.probs.top1)
            name = res.names.get(top1_idx, f"class_{top1_idx}")
            # Find accident class probability
            acc_idx = 1
            for k, v in res.names.items():
                if "accident" in v.lower() and "non" not in v.lower():
                    acc_idx = k
                    break
            acc_prob = float(res.probs.data[acc_idx].cpu().numpy())
            return (acc_prob, name)
        except Exception as exc:
            log.warning("Accident classification inference error: %s", exc)
            return None

    # ── Inference ──────────────────────────────────────────────────────────

    def _preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Validates and normalises an input frame for YOLO inference."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        import cv2
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3:
            ch = frame.shape[2]
            if ch == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif ch == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif ch != 3:
                return None
        else:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        return frame

    def detect(
        self,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
    ) -> List[Detection]:
        """
        Runs object detection on a single BGR frame.

        Args:
            frame:        BGR numpy array from OpenCV.
            frame_number: Optional frame index used for debug logging.

        Returns:
            List of Detection objects for NexGuard target classes only.
        """
        if not self.model_ready or self.model is None:
            return []

        processed = self._preprocess(frame)
        if processed is None:
            log.warning("Invalid frame for inference (frame %s) — skipping.", frame_number)
            return []

        t0 = time.monotonic()
        with self._lock:
            try:
                results = self.model(
                    processed,
                    verbose=False,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                )[0]
            except Exception as exc:
                err = type(exc).__name__
                if "out of memory" in str(exc).lower() or "OutOfMemory" in err:
                    log.error("Inference OOM on device '%s' — frame skipped.", self.device)
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    log.warning("Inference error frame %s: %s (%s)", frame_number, exc, err)
                return []
            finally:
                self.last_inference_time_ms = (time.monotonic() - t0) * 1000.0

        detections: List[Detection] = []
        target_classes = set(cfg.all_target_classes)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            is_acc = (
                cls_name in cfg.accident_classes
                or any(a.lower() in cls_name.lower() for a in cfg.accident_classes)
            )
            if cls_name not in target_classes and not is_acc:
                continue  # skip irrelevant classes (e.g. 'apple', 'vase')

            xyxy = box.xyxy[0].cpu().numpy()
            conf_val = float(box.conf[0].cpu().numpy())
            log.debug(
                "Frame %s | %s (%.2f) | box=[%.0f,%.0f,%.0f,%.0f] | %.1fms",
                frame_number, cls_name, conf_val,
                float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]),
                self.last_inference_time_ms,
            )
            detections.append(Detection(
                bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                class_name=cls_name,
                confidence=conf_val,
                class_id=cls_id,
                is_accident=is_acc,
            ))

        return detections

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def class_names(self) -> dict:
        return self._class_names

    @property
    def is_accident_model(self) -> bool:
        if not self.diagnostic_info:
            return False
        return self.diagnostic_info.model_type in (
            ModelType.ACCIDENT_DETECTOR,
            ModelType.HYBRID_DETECTOR,
        )

    def get_diagnostic_info(self) -> ModelDiagnosticInfo:
        if self.diagnostic_info is not None:
            return self.diagnostic_info
        return ModelDiagnosticInfo(
            model_path=self.model_path,
            device=self.device,
            model_type=ModelType.UNKNOWN,
            class_count=0,
            classes={},
            accident_classes_found=[],
            inference_size=self.imgsz,
            confidence_threshold=self.conf,
            dataset_status="UNKNOWN",
            ready=False,
        )

    def __repr__(self) -> str:
        status = "READY" if self.model_ready else "FAILED"
        return (
            f"ObjectDetector(model={self.model_path!r}, "
            f"device={self.device!r}, conf={self.conf}, status={status})"
        )
