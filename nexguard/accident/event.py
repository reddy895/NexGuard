"""
NexGuard — Incident Lifecycle Manager
=======================================
Manages the full lifecycle of confirmed accident incidents from
detection through resolution.

Lifecycle:
    DETECTED → CONFIRMED → ALERT_SENT → MONITORING → RESOLVED
"""

from __future__ import annotations

import math
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
        self._last_global_alert_time: float = 0.0
        self._frame_index: int = 0

    def _find_active_match(
        self, track_ids: List[int], location: Dict[str, float]
    ) -> Optional[Incident]:
        """Finds an existing active incident matching the tracks or spatial location."""
        track_set = set(track_ids)
        cx, cy = location.get("cx", 0.0), location.get("cy", 0.0)

        for inc in self._active.values():
            # Match 1: Overlapping involved tracks
            if track_set.intersection(inc.involved_tracks):
                return inc
            # Match 2: Proximity within 150 pixels of existing incident center
            inc_cx = inc.location.get("cx", 0.0)
            inc_cy = inc.location.get("cy", 0.0)
            if math.hypot(cx - inc_cx, cy - inc_cy) < 150.0:
                return inc
        return None

    def create_incident(
        self,
        event: AccidentEvent,
        severity_result: SeverityResult,
        involved_tracks: List[Track],
        person_involvements: Optional[List[PersonInvolvement]] = None,
    ) -> Incident:
        """
        Creates a new Incident from a confirmed AccidentEvent or updates
        an existing active incident to prevent duplicate alerts.

        Args:
            event:            Confirmed AccidentEvent.
            severity_result:  Computed severity.
            involved_tracks:  Track objects involved in the event.
            person_involvements: Optional person involvement records.

        Returns:
            Created or updated Incident object.
        """
        existing = self._find_active_match(event.track_ids, event.location)
        if existing is not None:
            # Deduplicate: update active incident instead of creating a new one
            existing.confidence = max(existing.confidence, event.confidence)
            if severity_result.score > existing.severity_score:
                existing.severity = severity_result.severity
                existing.severity_score = severity_result.score
                existing.severity_reasons = severity_result.reasons
            # Merge involved tracks and vehicles
            existing.involved_tracks = list(set(existing.involved_tracks + event.track_ids))
            new_veh = [t.track_id for t in involved_tracks if t.is_vehicle]
            existing.involved_vehicles = list(set(existing.involved_vehicles + new_veh))
            existing.frame_index = event.frame_index
            for sig in event.signals:
                if sig not in existing.signals:
                    existing.signals.append(sig)
            log.info(
                f"[INCIDENT DEDUPLICATED] Updated active incident {existing.incident_id} "
                f"(tracks={existing.involved_tracks})"
            )
            return existing

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
        - Initial alert (CONFIRMED and not yet sent)
        - Cooldown between repeat alerts for the same incident
        - Minimum global alert spacing to prevent flood
        """
        now = time.time()
        last = self._last_alert_times.get(incident.incident_id, 0.0)

        # First alert dispatch
        if incident.status == IncidentStatus.CONFIRMED and incident.alert_status != "SENT":
            # Avoid sending rapid bursts within 3s across any incident
            if (now - self._last_global_alert_time) < 3.0 and incident.severity != Severity.CRITICAL:
                return False
            return True

        # Re-alerting cooldown enforcement for existing incident
        if (now - last) >= self._cooldown_s:
            # Escalation — only re-alert if severity is HIGH or CRITICAL
            if incident.severity in (Severity.HIGH, Severity.CRITICAL):
                return True

        return False

    def mark_alert_sent(self, incident: Incident) -> None:
        """Updates incident status after alert dispatch."""
        now = time.time()
        incident.mark_alert_sent()
        incident.status = IncidentStatus.MONITORING
        self._last_alert_times[incident.incident_id] = now
        self._last_global_alert_time = now
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
