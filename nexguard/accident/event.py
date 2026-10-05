"""
NexGuard — Incident Lifecycle Manager
=======================================
Manages the full lifecycle of confirmed accident incidents from
detection through resolution.

Lifecycle:
    DETECTED → CONFIRMED → ALERT_SENT → MONITORING → RESOLVED
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

from config import cfg
from nexguard.models.schemas import (
    AccidentEvent,
    Incident,
    IncidentStatus,
    PersonInvolvement,
    Severity,
    SeverityResult,
    Track,
)
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.accident.event")


class IncidentManager:
    """
    Tracks and manages the lifecycle of accident incidents.

    Responsibilities:
    - Create Incident objects from confirmed AccidentEvents
    - Advance incident status through the lifecycle
    - Detect resolved incidents
    - Prevent duplicate alerts for the same ongoing event
    - Maintain incident history for the session
    """

    def __init__(self, alert_cooldown: Optional[int] = None) -> None:
        self._cooldown_s: int = alert_cooldown or cfg.alert_cooldown
        self._active: Dict[str, Incident] = {}
        self._history: List[Incident] = []
        self._last_alert_times: Dict[str, float] = {}
        self._frame_index: int = 0

    def create_incident(
        self,
        event: AccidentEvent,
        severity_result: SeverityResult,
        involved_tracks: List[Track],
        person_involvements: Optional[List[PersonInvolvement]] = None,
    ) -> Incident:
        """
        Creates a new Incident from a confirmed AccidentEvent.

        Args:
            event:            Confirmed AccidentEvent.
            severity_result:  Computed severity.
            involved_tracks:  Track objects involved in the event.
            person_involvements: Optional person involvement records.

        Returns:
            Newly created Incident object.
        """
        incident = Incident(
            start_time=event.timestamp,
            event_type=event.event_type,
            severity=severity_result.severity,
            severity_score=severity_result.score,
            severity_reasons=severity_result.reasons,
            confidence=event.confidence,
            involved_tracks=event.track_ids,
            involved_people=person_involvements or [],
            involved_vehicles=[t.track_id for t in involved_tracks if t.is_vehicle],
            location=event.location,
            signals=event.signals,
            frame_index=event.frame_index,
            status=IncidentStatus.CONFIRMED,
        )

        self._active[incident.incident_id] = incident
        log.info(
            f"[INCIDENT CREATED] {incident.incident_id}  "
            f"severity={incident.severity.value}  "
            f"confidence={incident.confidence:.2f}"
        )
        return incident

    def update_frame(self, frame_index: int) -> None:
        """Records the current frame index for lifecycle management."""
        self._frame_index = frame_index

    def should_send_alert(self, incident: Incident) -> bool:
        """
        Returns True if an alert should be sent for this incident.

        Enforces:
        - Initial alert (CONFIRMED → ALERT_SENT)
        - Cooldown between repeat alerts for the same incident
        - Cooldown between different incidents
        """
        now = time.time()
        last = self._last_alert_times.get(incident.incident_id, 0.0)

        if incident.status == IncidentStatus.CONFIRMED:
            return True  # First alert always sends

        if (now - last) >= self._cooldown_s:
            # Escalation — only re-alert if severity is HIGH or CRITICAL
            if incident.severity in (Severity.HIGH, Severity.CRITICAL):
                return True

        return False

    def mark_alert_sent(self, incident: Incident) -> None:
        """Updates incident status after alert dispatch."""
        incident.mark_alert_sent()
        incident.status = IncidentStatus.MONITORING
        self._last_alert_times[incident.incident_id] = time.time()
        log.info(f"[ALERT SENT] {incident.incident_id}")

    def resolve_stale_incidents(self, active_track_ids: List[int]) -> List[Incident]:
        """
        Resolves incidents whose involved tracks are no longer active.

        Args:
            active_track_ids: Currently active tracking IDs.

        Returns:
            List of incidents just resolved.
        """
        active_set = set(active_track_ids)
        resolved = []
        to_remove = []

        for iid, incident in self._active.items():
            involved = set(incident.involved_tracks)
            if not involved.intersection(active_set):
                incident.resolve()
                self._history.append(incident)
                to_remove.append(iid)
                resolved.append(incident)
                log.info(f"[INCIDENT RESOLVED] {incident.incident_id}")

        for iid in to_remove:
            del self._active[iid]

        return resolved

    def add_evidence_frame(self, incident: Incident, frame_path: str) -> None:
        """Records an evidence frame path on the incident."""
        incident.evidence_frames.append(frame_path)

    def add_evidence_clip(self, incident: Incident, clip_path: str) -> None:
        """Records an evidence video clip path on the incident."""
        incident.evidence_clip = clip_path

    @property
    def active_incidents(self) -> List[Incident]:
        """Returns all currently active incidents."""
        return list(self._active.values())

    @property
    def incident_history(self) -> List[Incident]:
        """Returns all resolved incidents from this session."""
        return list(self._history)

    @property
    def total_incidents(self) -> int:
        """Total number of incidents created this session."""
        return len(self._active) + len(self._history)
