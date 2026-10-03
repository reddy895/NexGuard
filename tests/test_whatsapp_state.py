"""
Test Suite 5: WhatsApp Session State & Cooldown Tests
"""

import pytest
import time
from whatsapp.client import WhatsAppClient


def test_whatsapp_disconnected_graceful_handling():
    client = WhatsAppClient()
    # Mock status check as disconnected
    client.check_status = lambda: {"status": "NOT CONNECTED", "connected": False}

    res = client.send_accident_alert(
        incident_id="NG-TEST-9999",
        severity_level="HIGH",
        confidence=0.85,
        detected_summary="2 Cars",
        timestamp="2026-10-02 12:00:00"
    )

    assert res["sent"] is False
    assert res["status"] == "NOT CONNECTED"
    assert "unavailable" in res["reason"].lower()


def test_whatsapp_cooldown_debouncing():
    client = WhatsAppClient()
    client.cached_status = {"status": "CONNECTED", "connected": True}
    client.cooldown_seconds = 300
    client.last_alert_time = time.time()  # Just sent alert

    res = client.send_accident_alert(
        incident_id="NG-TEST-9998",
        severity_level="CRITICAL",
        confidence=0.92,
        detected_summary="1 Truck, 1 Person",
        timestamp="2026-10-02 12:00:10"
    )

    assert res["sent"] is False
    assert res["status"] == "COOLDOWN_ACTIVE"
