"""
NexGuard Incident Management Module
Tracks active incidents, enforces alert cooldown periods, and formats incident records.
"""

import uuid
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from config import config
from utils.logger import logger
from src.evidence_manager import EvidenceManager


@dataclass
class IncidentRecord:
    incident_id: str
    timestamp: str
    source_type: str
    severity_level: str
    confidence_score: float
    involved_track_ids: List[int]
    people_near: int
    reasons: List[str]
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "source_type": self.source_type,
            "severity_level": self.severity_level,
            "confidence_score": round(self.confidence_score, 4),
            "involved_track_ids": self.involved_track_ids,
            "people_near": self.people_near,
            "reasons": self.reasons,
            "image_path": self.image_path
        }


class IncidentManager:
    """Manages active incidents and prevents redundant alert generation."""

    def __init__(self, cooldown_seconds: int = None):
        self.cooldown_seconds = cooldown_seconds or config.accident_cooldown_seconds
        self.evidence_manager = EvidenceManager()
        self.last_incident_time: float = 0.0
        self.active_incident: Optional[IncidentRecord] = None
        self.incident_history: List[IncidentRecord] = []

    def can_trigger_incident(self) -> bool:
        return (time.time() - self.last_incident_time) >= self.cooldown_seconds

    def record_incident(
        self,
        frame: np.ndarray,
        severity_level: str,
        confidence_score: float,
        involved_track_ids: List[int],
        people_near: int,
        reasons: List[str],
        source_type: str = "Video"
    ) -> Optional[IncidentRecord]:
        """Creates incident record, saves evidence frame, and initiates cooldown."""
        if not self.can_trigger_incident():
            logger.info("Incident alert suppressed due to active cooldown.")
            return self.active_incident

        inc_id = str(uuid.uuid4())[:8]
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")

        rec = IncidentRecord(
            incident_id=inc_id,
            timestamp=ts_str,
            source_type=source_type,
            severity_level=severity_level,
            confidence_score=confidence_score,
            involved_track_ids=involved_track_ids,
            people_near=people_near,
            reasons=reasons
        )

        img_path, json_path = self.evidence_manager.save_evidence(
            frame=frame,
            incident_id=inc_id,
            metadata=rec.to_dict()
        )

        if img_path:
            rec.image_path = str(img_path)

        self.last_incident_time = time.time()
        self.active_incident = rec
        self.incident_history.append(rec)
        logger.info(f"Recorded new incident [{inc_id}] - Severity: {severity_level}")
        return rec

    def reset_cooldown(self):
        """Resets incident cooldown timer."""
        self.last_incident_time = 0.0
        self.active_incident = None
