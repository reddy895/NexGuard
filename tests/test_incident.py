"""
Test Suite 4: Incident Storage & Evidence Capture Tests
"""

import pytest
import numpy as np
from utils.incident_logger import incident_store, IncidentRecord


def test_incident_record_creation():
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rec = incident_store.record_incident(
        frame=dummy_frame,
        severity_level="HIGH",
        confidence=0.88,
        detected_objects={"car": 2, "person": 1},
        source_type="Test"
    )

    assert rec.incident_id.startswith("NG-")
    assert rec.severity == "HIGH"
    assert rec.confidence == 0.88
    assert "car" in rec.detected_objects
