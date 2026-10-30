"""
Unit Tests for NexGuard Alert Rate Limiting and Duplicate Prevention
====================================================================
"""

import time
from pathlib import Path
import pytest
from nexguard.alerts.deduplicator import AlertRateLimiter
from nexguard.models.schemas import Incident, Severity, EventType

def test_alert_limiter_severity_gating():
    limiter = AlertRateLimiter(min_alert_severity=Severity.HIGH)

    low_inc = Incident(
        incident_id="INC-1",
        severity=Severity.LOW,
        confidence=0.90,
    )
    decision = limiter.evaluate_incident(low_inc)
    assert decision.allow_alert is False
    assert "below alert threshold" in decision.reason

    high_inc = Incident(
        incident_id="INC-2",
        severity=Severity.HIGH,
        confidence=0.85,
    )
    decision2 = limiter.evaluate_incident(high_inc)
    assert decision2.allow_alert is True

def test_alert_limiter_incident_cooldown():
    limiter = AlertRateLimiter(
        incident_cooldown_s=10.0,
        global_throttle_s=0.0,
        min_alert_severity=Severity.LOW,
    )

    inc = Incident(
        incident_id="INC-3",
        severity=Severity.MEDIUM,
        confidence=0.75,
        involved_tracks=[1, 2],
    )

    # First alert allowed
    d1 = limiter.evaluate_incident(inc)
    assert d1.allow_alert is True

    # Immediate second alert for same incident suppressed
    d2 = limiter.evaluate_incident(inc)
    assert d2.allow_alert is False
    assert "already alerted" in d2.reason

def test_alert_limiter_track_signature_dedup():
    limiter = AlertRateLimiter(
        incident_cooldown_s=10.0,
        global_throttle_s=0.0,
        min_alert_severity=Severity.LOW,
    )

    inc_a = Incident(
        incident_id="INC-A",
        severity=Severity.MEDIUM,
        confidence=0.75,
        involved_tracks=[10, 20],
    )
    d1 = limiter.evaluate_incident(inc_a)
    assert d1.allow_alert is True

    # Different incident ID but identical vehicles involved
    inc_b = Incident(
        incident_id="INC-B",
        severity=Severity.HIGH,
        confidence=0.85,
        involved_tracks=[20, 10],  # reverse order
    )
    d2 = limiter.evaluate_incident(inc_b)
    assert d2.allow_alert is False
    assert "already in active alert cooldown" in d2.reason
