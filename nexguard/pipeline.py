"""
NexGuard — End-to-End Modular Inference Pipeline
================================================
Orchestrates the full multi-stage accident detection workflow:
  Frame Preprocessing
       ↓
  Object Detection (YOLOv8)
       ↓
  Object Tracking (Spatial & IoU)
       ↓
  Temporal Accident Analysis (Kinematics & Multi-frame aggregation)
       ↓
  Accident Verification & Calibration
       ↓
  Severity Classification & Evidence Extraction
       ↓
  Incident Lifecycle Management & Alerting
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from config import cfg
from nexguard.accident.accident_detector import AccidentDetector
from nexguard.accident.calibration import ConfidenceCalibrator
from nexguard.accident.event import IncidentManager
from nexguard.accident.severity import SeverityClassifier, analyze_person_involvement
from nexguard.accident.suppression import FalsePositiveFilter
from nexguard.detection.detector import ObjectDetector
from nexguard.models.schemas import (
    AccidentEvent,
    Detection,
    Incident,
    SeverityResult,
    Track,
)
from nexguard.tracking.tracker import ObjectTracker
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.pipeline")

@dataclass
class FramePipelineResult:
    frame_index: int
    detections: List[Detection]
    active_tracks: List[Track]
    accident_event: Optional[AccidentEvent]
    severity_result: Optional[SeverityResult]
    incident: Optional[Incident]
    inference_latency_ms: float
    fps: float
    model_accident_prob: Optional[float] = None

class InferencePipeline:
    """
    Modular end-to-end inference engine for live video feeds and files.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        tracker: Optional[ObjectTracker] = None,
        accident_detector: Optional[AccidentDetector] = None,
        severity_classifier: Optional[SeverityClassifier] = None,
        incident_manager: Optional[IncidentManager] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ) -> None:
        self.detector = detector or ObjectDetector()
        self.tracker = tracker or ObjectTracker()
        self.accident_detector = accident_detector or AccidentDetector()
        self.severity_classifier = severity_classifier or SeverityClassifier()
        self.incident_manager = incident_manager or IncidentManager()
        self.calibrator = calibrator or ConfidenceCalibrator()

        self.frame_count: int = 0
        self._last_fps_time = time.monotonic()
        self.current_fps: float = 0.0

    def process_frame(self, frame: np.ndarray) -> FramePipelineResult:
        t0 = time.perf_counter()
        self.frame_count += 1

        # 1. Object Detection
        detections = self.detector.detect(frame, frame_number=self.frame_count)

        # 2. Direct Model Accident Prior (if classifier is loaded)
        cls_result = self.detector.classify_accident(frame)
        model_acc_prob = cls_result[0] if cls_result else None

        # 3. Object Tracking
        tracks = self.tracker.update(detections)

        # 4. Temporal Accident Analysis
        raw_event = self.accident_detector.update(tracks)
        confirmed_event: Optional[AccidentEvent] = None

        if raw_event and raw_event.accident_detected:
            # Calibrate confidence
            calibrated_conf = self.calibrator.calibrate(raw_event.confidence)
            raw_event.confidence = calibrated_conf
            confirmed_event = raw_event

        # 5. Severity Assessment & Incident Lifecycle
        severity_res: Optional[SeverityResult] = None
        incident: Optional[Incident] = None

        if confirmed_event:
            involved = [t for t in tracks if t.track_id in confirmed_event.track_ids]
            persons = [t for t in tracks if t.is_person]
            involvements = analyze_person_involvement(persons, confirmed_event)

            # Update or create incident
            incident, is_new = self.incident_manager.get_or_create(confirmed_event, frame)
            severity_res = self.severity_classifier.classify(
                event=confirmed_event,
                involved_tracks=involved,
                all_tracks=tracks,
                person_involvements=involvements,
                persistence_frames=incident.persistence_frames if incident else 0,
            )
            if incident:
                incident.severity = severity_res.severity
                incident.evidence_reasons = severity_res.reasons

        # Resolve stale incidents
        active_ids = [t.track_id for t in tracks]
        self.incident_manager.resolve_stale(active_ids)

        latency = (time.perf_counter() - t0) * 1000.0

        # Update FPS
        now = time.monotonic()
        dt = now - self._last_fps_time
        if dt >= 0.5:
            self.current_fps = 1.0 / max(dt, 1e-6)
            self._last_fps_time = now

        return FramePipelineResult(
            frame_index=self.frame_count,
            detections=detections,
            active_tracks=tracks,
            accident_event=confirmed_event,
            severity_result=severity_res,
            incident=incident,
            inference_latency_ms=round(latency, 2),
            fps=round(self.current_fps, 1),
            model_accident_prob=model_acc_prob,
        )

    def reset(self) -> None:
        self.tracker.reset()
        self.accident_detector.reset()
        self.incident_manager.reset()
        self.frame_count = 0
