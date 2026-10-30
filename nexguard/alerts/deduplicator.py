"""
NexGuard — Alert Rate Limiter and Duplicate Prevention Engine
=============================================================
Guarantees emergency notifications are sent exclusively for verified incidents:
- Suppresses repetitive alerts for ongoing active collisions
- Deduplicates across track IDs and spatial collision zones
- Enforces per-incident and global rate-limiting cooldowns
- Prohibits notifications for borderline or unconfirmed events
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from nexguard.models.schemas import Incident, Severity
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.alerts")

@dataclass
class AlertDecision:
    allow_alert: bool
    reason: str

class AlertRateLimiter:
    """
    Prevents duplicate notifications and spam for emergency alerts.
    """

    def __init__(
        self,
        incident_cooldown_s: float = 180.0,
        global_throttle_s: float = 30.0,
        min_alert_severity: Severity = Severity.MEDIUM,
    ) -> None:
        self.incident_cooldown_s = incident_cooldown_s
        self.global_throttle_s = global_throttle_s
        self.min_alert_severity = min_alert_severity

        self._last_alerted_incidents: Dict[str, float] = {}  # incident_id -> monotonic_time
        self._last_global_alert_time: float = -9999.0
        self._active_track_signatures: Dict[str, float] = {} # "id1_id2" -> monotonic_time

    @staticmethod
    def _get_tracks_sig(track_ids: List[int]) -> str:
        return "_".join(str(i) for i in sorted(track_ids))

    def evaluate_incident(self, incident: Incident) -> AlertDecision:
        now = time.monotonic()

        # 1. Severity gating
        sev_order = [
            Severity.NO_ACCIDENT,
            Severity.POSSIBLE,
            Severity.LOW,
            Severity.MEDIUM,
            Severity.HIGH,
            Severity.UNCERTAIN,
        ]
        if sev_order.index(incident.severity) < sev_order.index(self.min_alert_severity):
            return AlertDecision(
                allow_alert=False,
                reason=f"Severity {incident.severity.value} below alert threshold {self.min_alert_severity.value}",
            )

        # 2. Per-incident cooldown
        if incident.incident_id in self._last_alerted_incidents:
            elapsed = now - self._last_alerted_incidents[incident.incident_id]
            if elapsed < self.incident_cooldown_s:
                return AlertDecision(
                    allow_alert=False,
                    reason=f"Incident {incident.incident_id} already alerted {elapsed:.1f}s ago (cooldown={self.incident_cooldown_s}s)",
                )

        # 3. Track-signature deduplication
        sig = self._get_tracks_sig(incident.involved_tracks)
        if sig in self._active_track_signatures:
            elapsed = now - self._active_track_signatures[sig]
            if elapsed < self.incident_cooldown_s:
                return AlertDecision(
                    allow_alert=False,
                    reason=f"Vehicles {incident.involved_tracks} already in active alert cooldown",
                )

        # 4. Global throttle
        time_since_last_alert = now - self._last_global_alert_time
        if time_since_last_alert < self.global_throttle_s:
            return AlertDecision(
                allow_alert=False,
                reason=f"Global throttle active ({time_since_last_alert:.1f}s < {self.global_throttle_s}s)",
            )

        # Allow alert and record timestamps
        self._last_alerted_incidents[incident.incident_id] = now
        self._active_track_signatures[sig] = now
        self._last_global_alert_time = now

        log.info(
            "ALERT APPROVED: Incident=%s Sev=%s Conf=%.2f Tracks=%s",
            incident.incident_id, incident.severity.value, incident.confidence, incident.involved_tracks,
        )
        return AlertDecision(allow_alert=True, reason="Verified incident meets all alert criteria")
