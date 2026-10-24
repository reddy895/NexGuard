"""
Unit tests for AlertManager, WhatsAppNotifier, and Incident deduplication.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from config import cfg
from nexguard.accident.event import IncidentManager
from nexguard.alerts.manager import AlertManager
from nexguard.alerts.whatsapp import WhatsAppNotifier
from nexguard.models.schemas import (
    AccidentEvent,
    EventType,
    Incident,
    IncidentStatus,
    Severity,
    SeverityResult,
    Track,
)


def test_whatsapp_notifier_disabled():
    notifier = WhatsAppNotifier(enabled=False)
    assert not notifier.is_connected
    callback_called = []

    def on_done(success, status):
        callback_called.append((success, status))

    notifier.send("9591152862", "Test message", on_complete=on_done)
    assert len(callback_called) == 1
    assert callback_called[0][0] is False
    assert callback_called[0][1] == "DISABLED"


def test_whatsapp_notifier_offline_handling():
    notifier = WhatsAppNotifier(enabled=True)
    # Patch requests.post to simulate offline bot
    with patch("requests.post", side_effect=Exception("Connection refused")):
        res = notifier._send_sync("9591152862", "Test message")
        assert res is False
        assert notifier.total_failed >= 1
        history = notifier.get_delivery_history()
        assert len(history) >= 1
        assert history[-1]["success"] is False


def test_alert_manager_prevents_fake_sent_when_offline():
    mock_notifier = MagicMock(spec=WhatsAppNotifier)
    mock_notifier.enabled = True
    mock_notifier.is_connected = False
    mock_notifier.get_status.return_value = "OFFLINE"

    manager = AlertManager(notifier=mock_notifier)
    incident = Incident(
        incident_id="INC-TEST01",
        event_type=EventType.VEHICLE_COLLISION,
        severity=Severity.HIGH,
        severity_score=0.85,
        confidence=0.80,
        status=IncidentStatus.CONFIRMED,
    )

    sent = manager.dispatch(incident)
    assert sent is False
    assert "DISCONNECTED" in incident.alert_status
    mock_notifier.send.assert_not_called()


def test_alert_manager_dispatch_success_message_formatting():
    mock_notifier = MagicMock(spec=WhatsAppNotifier)
    mock_notifier.enabled = True
    mock_notifier.is_connected = True
    mock_notifier.get_status.return_value = "READY"

    manager = AlertManager(notifier=mock_notifier)
    incident = Incident(
        incident_id="INC-TEST02",
        event_type=EventType.VEHICLE_COLLISION,
        severity=Severity.HIGH,
        severity_score=0.88,
        confidence=0.82,
        involved_vehicles=[1, 2],
        evidence_frames=["/tmp/evidence_frame_001.jpg"],
        status=IncidentStatus.CONFIRMED,
    )

    sent = manager.dispatch(incident)
    assert sent is True
    assert mock_notifier.send.called

    # Verify message contents
    args, _ = mock_notifier.send.call_args
    phone, msg = args
    assert "INC-TEST02" in msg
    assert cfg.camera_id in msg
    assert "82%" in msg  # Confidence
    assert "evidence_frame_001.jpg" in msg
    assert "SAFETY NOTICE" in msg  # Disclaimer


def test_incident_deduplication():
    mgr = IncidentManager(alert_cooldown=60)
    event1 = AccidentEvent(
        accident_detected=True,
        confidence=0.75,
        timestamp="2026-10-24T12:00:00",
        track_ids=[10, 11],
        location={"cx": 100.0, "cy": 100.0},
        event_type=EventType.VEHICLE_COLLISION,
        signals=["bbox_overlap=0.25"],
        frame_index=1,
    )
    sev1 = SeverityResult(severity=Severity.MEDIUM, score=0.6, reasons=["Overlap"])
    t1 = Track(track_id=10, bbox=(90, 90, 110, 110), class_name="car", confidence=0.8, is_vehicle=True)
    t2 = Track(track_id=11, bbox=(95, 95, 115, 115), class_name="car", confidence=0.8, is_vehicle=True)

    inc1 = mgr.create_incident(event1, sev1, [t1, t2])
    assert inc1.incident_id is not None
    assert len(mgr.active_incidents) == 1

    # Second event for same vehicles
    event2 = AccidentEvent(
        accident_detected=True,
        confidence=0.85,
        timestamp="2026-10-24T12:00:01",
        track_ids=[10, 11],
        location={"cx": 102.0, "cy": 101.0},
        event_type=EventType.VEHICLE_COLLISION,
        signals=["speed_drop=50%"],
        frame_index=2,
    )
    sev2 = SeverityResult(severity=Severity.HIGH, score=0.85, reasons=["Speed drop"])

    inc2 = mgr.create_incident(event2, sev2, [t1, t2])
    # Should deduplicate to the same incident ID
    assert inc2.incident_id == inc1.incident_id
    assert len(mgr.active_incidents) == 1
    assert inc2.confidence == 0.85
    assert inc2.severity == Severity.HIGH


def test_incident_cooldown_enforcement():
    mgr = IncidentManager(alert_cooldown=30)
    incident = Incident(
        incident_id="INC-COOLDOWN",
        status=IncidentStatus.CONFIRMED,
        severity=Severity.HIGH,
    )

    # First alert should be allowed
    assert mgr.should_send_alert(incident) is True

    # Mark alert sent
    mgr.mark_alert_sent(incident)
    assert incident.alert_status == "SENT"
    assert incident.status == IncidentStatus.MONITORING

    # Immediate second call should be blocked by cooldown
    assert mgr.should_send_alert(incident) is False


def test_dispatch_is_non_blocking():
    # Verify that a slow send does not block the caller
    mock_notifier = MagicMock(spec=WhatsAppNotifier)
    mock_notifier.enabled = True
    mock_notifier.is_connected = True

    def slow_send(*args, **kwargs):
        time.sleep(0.05)

    mock_notifier.send.side_effect = slow_send
    manager = AlertManager(notifier=mock_notifier)
    incident = Incident(
        incident_id="INC-FAST",
        severity=Severity.HIGH,
        status=IncidentStatus.CONFIRMED,
    )

    t0 = time.monotonic()
    manager.dispatch(incident)
    elapsed = time.monotonic() - t0
    # manager.dispatch should complete rapidly
    assert elapsed < 0.20
