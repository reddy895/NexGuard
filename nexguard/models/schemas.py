"""
NexGuard — Data Models and Schemas
====================================
Plain Python dataclasses used throughout the pipeline.
No ORM or external framework dependency required.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Accident severity levels displayed in the video overlay."""
    NO_ACCIDENT = "NO_ACCIDENT"
    POSSIBLE    = "POSSIBLE"
    LOW         = "LOW"
    MEDIUM      = "MEDIUM"
    HIGH        = "HIGH"
    UNCERTAIN   = "UNCERTAIN"


class EventType(str, Enum):
    """Type of accident / safety event detected."""
    VEHICLE_COLLISION    = "vehicle_collision"
    VEHICLE_PEDESTRIAN   = "vehicle_pedestrian"
    SUDDEN_STOP          = "sudden_stop"
    POSSIBLE_ACCIDENT    = "possible_accident"
    CONFIRMED_MODEL_EVENT = "confirmed_model_event"
    UNKNOWN              = "unknown"


class DetectionMode(str, Enum):
    """How the accident was detected."""
    HEURISTIC     = "heuristic"
    TRAINED_MODEL = "trained_model"


class ModelType(str, Enum):
    """Classification of the loaded YOLO model."""
    GENERIC_DETECTOR  = "generic_detector"
    ACCIDENT_DETECTOR = "accident_detector"
    HYBRID_DETECTOR   = "hybrid_detector"
    UNKNOWN           = "unknown"


class IncidentStatus(str, Enum):
    """Lifecycle status of an incident."""
    ACTIVE   = "ACTIVE"
    RESOLVED = "RESOLVED"


# ---------------------------------------------------------------------------
# Detection output
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Raw output from the object detector for one detected object.

    Attributes:
        bbox:       (x1, y1, x2, y2) in pixel coordinates.
        class_name: COCO label (e.g., 'car', 'person').
        confidence: Detection score in [0, 1].
        class_id:   Numeric class ID from the model.
        is_accident: True when the class is an accident-specific label.
    """
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    class_id: int = 0
    is_accident: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    @property
    def area(self) -> float:
        return max(0.0, (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


# ---------------------------------------------------------------------------
# Tracked object
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """
    An object maintained across multiple frames by the tracker.

    Attributes:
        track_id:     Unique tracking ID assigned by the tracker.
        bbox:         Current (x1, y1, x2, y2) bounding box.
        class_name:   Object class label.
        confidence:   Latest detection confidence.
        cx, cy:       Current centroid.
        velocity:     (vx, vy) in pixels/frame.
        speed:        Scalar speed in pixels/frame.
        heading:      Direction of movement in degrees.
        speed_history: Recent speed samples (for deceleration analysis).
        history:      Recent centroid positions (for trajectory drawing).
        lost_frames:  Frames since last matched detection.
        age:          Total frames this track has existed.
        hits:         Number of frames matched by a detection.
        is_vehicle:   True when class is a vehicle type.
        is_person:    True when class is 'person'.
        is_accident:  True when class is an accident-specific label.
    """
    track_id: int
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    cx: float = 0.0
    cy: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    speed: float = 0.0
    heading: float = 0.0
    speed_history: List[float] = field(default_factory=list)
    history: List[Tuple[float, float]] = field(default_factory=list)
    lost_frames: int = 0
    age: int = 0
    hits: int = 1
    is_vehicle: bool = False
    is_person: bool = False
    is_accident: bool = False

    # ── Computed properties ───────────────────────────────────────────────

    @property
    def stability_score(self) -> float:
        """Hit rate over the track's lifetime (0.0–1.0)."""
        return min(self.hits / max(self.age, 1), 1.0)

    @property
    def is_stable(self) -> bool:
        """A track is stable once it has ≥3 detections and ≥40% hit rate."""
        return self.hits >= 3 and self.stability_score >= 0.40

    @property
    def avg_speed(self) -> float:
        """Average speed over recent history."""
        return sum(self.speed_history) / len(self.speed_history) if self.speed_history else 0.0

    @property
    def speed_drop_ratio(self) -> float:
        """
        Fractional speed drop relative to recent average.
        Positive value means deceleration occurred.
        """
        if self.avg_speed < 0.001:
            return 0.0
        return max(0.0, (self.avg_speed - self.speed) / self.avg_speed)

    @property
    def acceleration(self) -> float:
        """Rate of speed change (pixels/frame²). Negative = deceleration."""
        if len(self.speed_history) < 2:
            return 0.0
        return self.speed_history[-1] - self.speed_history[-2]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id":       self.track_id,
            "class_name":     self.class_name,
            "confidence":     round(self.confidence, 3),
            "bbox":           [round(v, 1) for v in self.bbox],
            "center":         (round(self.cx, 1), round(self.cy, 1)),
            "speed":          round(self.speed, 2),
            "heading":        round(self.heading, 1),
            "lost_frames":    self.lost_frames,
            "age":            self.age,
            "hits":           self.hits,
            "stability":      round(self.stability_score, 2),
            "is_vehicle":     self.is_vehicle,
            "is_person":      self.is_person,
            "is_accident":    self.is_accident,
        }


# ---------------------------------------------------------------------------
# Accident event
# ---------------------------------------------------------------------------

@dataclass
class AccidentEvent:
    """
    Candidate accident event produced by AccidentDetector.

    This is NOT a confirmed incident — it requires temporal confirmation
    before being escalated to an Incident.
    """
    accident_detected: bool
    confidence: float
    timestamp: str
    track_ids: List[int]
    location: Dict[str, float]          # {"cx": ..., "cy": ...}
    event_type: EventType
    signals: List[str]                  # Human-readable evidence signals
    frame_index: int = 0
    detection_mode: DetectionMode = DetectionMode.HEURISTIC
    is_model_confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accident_detected": self.accident_detected,
            "confidence":        round(self.confidence, 3),
            "timestamp":         self.timestamp,
            "track_ids":         self.track_ids,
            "location":          self.location,
            "event_type":        self.event_type.value,
            "signals":           self.signals,
            "frame_index":       self.frame_index,
            "detection_mode":    self.detection_mode.value,
            "is_model_confirmed": self.is_model_confirmed,
        }


# ---------------------------------------------------------------------------
# Severity result
# ---------------------------------------------------------------------------

@dataclass
class SeverityResult:
    """
    Output from the severity classification engine.

    severity_score is a continuous value in [0, 1].
    severity is the human-readable category derived from config thresholds.
    """
    severity: Severity
    score: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "score":    round(self.score, 3),
            "reasons":  self.reasons,
        }


# ---------------------------------------------------------------------------
# Person involvement
# ---------------------------------------------------------------------------

@dataclass
class PersonInvolvement:
    """Observable CV evidence of a person near an accident zone."""
    track_id: int
    distance_px: float
    movement: str           # "stationary" | "moving" | "decelerating"
    involvement_level: str  # "LOW" | "MEDIUM" | "HIGH"
    position: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id":         self.track_id,
            "distance_px":      round(self.distance_px, 1),
            "movement":         self.movement,
            "involvement_level": self.involvement_level,
            "position":         self.position,
        }


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """
    Full lifecycle object for a confirmed accident incident.
    Created by IncidentManager when AccidentDetector confirms an event.
    """
    incident_id: str = field(
        default_factory=lambda: f"NG-{str(uuid.uuid4())[:4].upper()}"
    )
    start_frame: int = 0
    event_type: EventType = EventType.UNKNOWN
    severity: Severity = Severity.LOW
    severity_score: float = 0.0
    severity_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    involved_tracks: List[int] = field(default_factory=list)
    involved_people: List[PersonInvolvement] = field(default_factory=list)
    location: Dict[str, float] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)
    status: IncidentStatus = IncidentStatus.ACTIVE

    def resolve(self) -> None:
        self.status = IncidentStatus.RESOLVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id":     self.incident_id,
            "start_frame":     self.start_frame,
            "event_type":      self.event_type.value,
            "severity":        self.severity.value,
            "severity_score":  round(self.severity_score, 3),
            "severity_reasons": self.severity_reasons,
            "confidence":      round(self.confidence, 3),
            "involved_tracks": self.involved_tracks,
            "location":        self.location,
            "signals":         self.signals,
            "status":          self.status.value,
        }


# ---------------------------------------------------------------------------
# Model diagnostic info
# ---------------------------------------------------------------------------

@dataclass
class ModelDiagnosticInfo:
    """Startup metadata for the loaded YOLO model."""
    model_path: str
    device: str
    model_type: ModelType
    class_count: int
    classes: Dict[int, str]
    accident_classes_found: List[str]
    inference_size: int
    confidence_threshold: float
    dataset_status: str
    ready: bool = True
    error_message: Optional[str] = None
