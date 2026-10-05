"""
NexGuard — Data Models and Schemas
====================================
Defines structured data objects used across the NexGuard pipeline.
These are plain Python dataclasses — no external ORM dependency required.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Accident severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentStatus(str, Enum):
    """Lifecycle status of an incident."""
    DETECTED = "DETECTED"
    CONFIRMED = "CONFIRMED"
    ALERT_SENT = "ALERT_SENT"
    MONITORING = "MONITORING"
    RESOLVED = "RESOLVED"


class EventType(str, Enum):
    """Type of accident / safety event detected."""
    VEHICLE_COLLISION = "vehicle_collision"
    VEHICLE_PEDESTRIAN = "vehicle_pedestrian"
    PERSON_FALL = "person_fall"
    SUDDEN_STOP = "sudden_stop"
    NEAR_MISS = "near_miss"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Detection output
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Raw output from the object detector for a single detected object.

    Attributes:
        bbox:       Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        class_name: COCO class label (e.g., 'car', 'person').
        confidence: Detection confidence score in [0, 1].
        class_id:   Numeric class ID from the model.
    """
    bbox: Tuple[float, float, float, float]
    class_name: str
    confidence: float
    class_id: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


# ---------------------------------------------------------------------------
# Tracked object
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """
    A tracked object maintained across multiple frames.

    Attributes:
        track_id:       Unique tracking identifier.
        bbox:           Current bounding box (x1, y1, x2, y2).
        class_name:     Object class label.
        confidence:     Latest detection confidence.
        cx, cy:         Current center coordinates.
        velocity:       (vx, vy) in pixels/frame.
        speed:          Scalar speed magnitude in pixels/frame.
        heading:        Direction of movement in degrees.
        speed_history:  Recent speed samples for deceleration analysis.
        history:        Recent center positions for trajectory drawing.
        lost_frames:    Frames since last successful detection match.
        age:            Total number of frames this track has been alive.
        is_vehicle:     True if this track represents a vehicle.
        is_person:      True if this track represents a person.
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
    is_vehicle: bool = False
    is_person: bool = False

    @property
    def acceleration(self) -> float:
        """Rate of speed change (pixels/frame²). Negative = deceleration."""
        if len(self.speed_history) < 2:
            return 0.0
        return self.speed_history[-1] - self.speed_history[-2]

    @property
    def avg_speed(self) -> float:
        """Average speed over recent history."""
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)

    @property
    def speed_drop_ratio(self) -> float:
        """
        Fraction by which speed has dropped relative to recent average.
        Returns 0.0 if speed has not dropped.
        """
        if self.avg_speed < 0.001:
            return 0.0
        drop = (self.avg_speed - self.speed) / self.avg_speed
        return max(0.0, drop)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "center": (round(self.cx, 1), round(self.cy, 1)),
            "speed": round(self.speed, 2),
            "heading": round(self.heading, 1),
        }


# ---------------------------------------------------------------------------
# Accident event
# ---------------------------------------------------------------------------

@dataclass
class AccidentEvent:
    """
    Structured output from the accident detection engine.

    This is NOT a final incident — it is a candidate event requiring
    temporal confirmation before escalation.
    """
    accident_detected: bool
    confidence: float
    timestamp: str
    track_ids: List[int]
    location: Dict[str, float]          # {"cx": ..., "cy": ...}
    event_type: EventType
    signals: List[str]                  # Human-readable evidence signals
    frame_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accident_detected": self.accident_detected,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "track_ids": self.track_ids,
            "location": self.location,
            "event_type": self.event_type.value,
            "signals": self.signals,
            "frame_index": self.frame_index,
        }


# ---------------------------------------------------------------------------
# Severity result
# ---------------------------------------------------------------------------

@dataclass
class SeverityResult:
    """
    Output from the severity classification engine.

    severity_score is a continuous value in [0, 1]; severity is the
    human-readable category derived from configurable thresholds.
    """
    severity: Severity
    score: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "score": round(self.score, 3),
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Person involvement
# ---------------------------------------------------------------------------

@dataclass
class PersonInvolvement:
    """Observed computer-vision evidence of a person near an accident zone."""
    track_id: int
    distance_px: float
    movement: str           # "stationary", "moving", "falling"
    involvement_level: str  # "LOW", "MEDIUM", "HIGH"
    last_known_position: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "distance_px": round(self.distance_px, 1),
            "movement": self.movement,
            "involvement_level": self.involvement_level,
            "last_known_position": self.last_known_position,
        }


# ---------------------------------------------------------------------------
# Incident
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """
    Full lifecycle object for a confirmed accident incident.

    Statuses:
        DETECTED   → raw event identified
        CONFIRMED  → temporal confirmation passed
        ALERT_SENT → WhatsApp notification dispatched
        MONITORING → ongoing monitoring after initial alert
        RESOLVED   → no longer active
    """
    incident_id: str = field(default_factory=lambda: f"INC-{str(uuid.uuid4())[:8].upper()}")
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    event_type: EventType = EventType.UNKNOWN
    severity: Severity = Severity.LOW
    severity_score: float = 0.0
    severity_reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    involved_tracks: List[int] = field(default_factory=list)
    involved_people: List[PersonInvolvement] = field(default_factory=list)
    involved_vehicles: List[int] = field(default_factory=list)
    evidence_frames: List[str] = field(default_factory=list)
    evidence_clip: Optional[str] = None
    alert_status: str = "PENDING"
    status: IncidentStatus = IncidentStatus.DETECTED
    location: Dict[str, float] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)
    frame_index: int = 0

    def resolve(self) -> None:
        self.status = IncidentStatus.RESOLVED
        self.end_time = datetime.now().isoformat()

    def mark_alert_sent(self) -> None:
        self.status = IncidentStatus.ALERT_SENT
        self.alert_status = "SENT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "severity_score": round(self.severity_score, 3),
            "severity_reasons": self.severity_reasons,
            "confidence": round(self.confidence, 3),
            "involved_tracks": self.involved_tracks,
            "involved_people": [p.to_dict() for p in self.involved_people],
            "involved_vehicles": self.involved_vehicles,
            "evidence_frames": self.evidence_frames,
            "evidence_clip": self.evidence_clip,
            "alert_status": self.alert_status,
            "status": self.status.value,
            "location": self.location,
            "signals": self.signals,
            "frame_index": self.frame_index,
        }
