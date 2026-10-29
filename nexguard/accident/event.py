"""
NexGuard — Incident Lifecycle Manager
=======================================
Manages the full lifecycle of confirmed accident incidents
from detection through resolution.

Lifecycle:
    AccidentEvent → create_incident() → Incident (ACTIVE)
                                      → resolve_stale_incidents() → RESOLVED
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from config import cfg
from nexguard.models.schemas import (
    AccidentEvent,
    Incident,
    IncidentStatus,
    PersonInvolvement,
    SeverityResult,
    Track,
)
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.event")


class IncidentManager:
    """
    Tracks and manages the lifecycle of accident incidents.

    Responsibilities:
    - Create Incident objects from confirmed AccidentEvents
    - Deduplicate concurrent events for the same collision
    - Advance incident status through the lifecycle
    - Resolve incidents when involved tracks leave the scene
    - Maintain bounded incident history for the session
    """

    _MAX_HISTORY = 200  # cap history to avoid unbounded memory growth

    def __init__(self) -> None:
        self._active: Dict[str, Incident] = {}
        self._history: List[Incident] = []
        self._frame_index: int = 0

    # ── Public interface ───────────────────────────────────────────────────

    def reset(self) -> None:
        """Resets active incidents and history for a new session."""
        self._active.clear()
        self._history.clear()
        self._frame_index = 0

    def update_frame(self, frame_index: int) -> None:
        """Records the current frame index for lifecycle management."""
        self._frame_index = frame_index

    def create_incident(
        self,
        event: AccidentEvent,
        severity_result: SeverityResult,
        involved_tracks: List[Track],
        person_involvements: Optional[List[PersonInvolvement]] = None,
    ) -> Incident:
        """
        Creates a new Incident from a confirmed AccidentEvent, or updates
        an existing active incident to prevent duplicate alerts.

        Args:
            event:               Confirmed AccidentEvent from the detector.
            severity_result:     Computed severity from SeverityClassifier.
            involved_tracks:     Track objects involved in the event.
            person_involvements: Person involvement records (optional).

        Returns:
            Created or updated Incident object.
        """
        existing = self._find_active_match(event.track_ids, event.location)
        if existing is not None:
            # Deduplicate: update active incident
            existing.confidence = max(existing.confidence, event.confidence)
            if severity_result.score > existing.severity_score:
                existing.severity = severity_result.severity
                existing.severity_score = severity_result.score
                existing.severity_reasons = severity_result.reasons
            existing.involved_tracks = list(set(existing.involved_tracks + event.track_ids))
            for sig in event.signals:
                if sig not in existing.signals:
                    existing.signals.append(sig)
            log.info("Incident %s updated (tracks=%s)", existing.incident_id, existing.involved_tracks)
            return existing

        incident = Incident(
            start_frame=event.frame_index,
            event_type=event.event_type,
            severity=severity_result.severity,
            severity_score=severity_result.score,
            severity_reasons=severity_result.reasons,
            confidence=event.confidence,
            involved_tracks=list(event.track_ids),
            involved_people=person_involvements or [],
            location=dict(event.location),
            signals=list(event.signals),
            status=IncidentStatus.ACTIVE,
        )
        self._active[incident.incident_id] = incident
        log.warning(
            "Incident %s CREATED — severity=%s conf=%.2f tracks=%s",
            incident.incident_id, incident.severity.value,
            incident.confidence, incident.involved_tracks,
        )
        return incident

    def resolve_stale_incidents(self, active_track_ids: List[int]) -> List[Incident]:
        """
        Resolves incidents whose involved tracks are no longer active.

        Args:
            active_track_ids: Currently active tracking IDs.

        Returns:
            List of incidents just resolved.
        """
        active_set = set(active_track_ids)
        resolved: List[Incident] = []
        to_remove: List[str] = []

        for iid, incident in self._active.items():
            if not set(incident.involved_tracks).intersection(active_set):
                incident.resolve()
                to_remove.append(iid)
                resolved.append(incident)
                log.info("Incident %s RESOLVED", iid)

        for iid in to_remove:
            self._history.append(self._active.pop(iid))

        # Enforce history cap
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

        return resolved

    # ── Properties ─────────────────────────────────────────────────────────

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
        """Total incidents created this session (active + resolved)."""
        return len(self._active) + len(self._history)

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_active_match(
        self,
        track_ids: List[int],
        location: Dict[str, float],
    ) -> Optional[Incident]:
        """Finds an existing active incident matching the tracks or location."""
        track_set = set(track_ids)
        cx, cy = location.get("cx", 0.0), location.get("cy", 0.0)
        for inc in self._active.values():
            if track_set.intersection(inc.involved_tracks):
                return inc
            inc_cx = inc.location.get("cx", 0.0)
            inc_cy = inc.location.get("cy", 0.0)
            if math.hypot(cx - inc_cx, cy - inc_cy) < 150.0:
                return inc
        return None
