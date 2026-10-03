"""
Unit Tests for NexGuard Incident Manager and Evidence Manager
"""

import pytest
import numpy as np
from pathlib import Path
from src.incident_manager import IncidentManager


def test_incident_record_and_cooldown(tmp_path):
    mgr = IncidentManager(cooldown_seconds=30)
    mgr.evidence_manager.target_dir = tmp_path

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    inc1 = mgr.record_incident(
        frame=dummy_frame,
        severity_level="HIGH",
        confidence_score=0.85,
        involved_track_ids=[1, 2],
        people_near=1,
        reasons=["Head-on collision"],
        source_type="TestVideo"
    )

    assert inc1 is not None
    assert inc1.severity_level == "HIGH"
    assert inc1.involved_track_ids == [1, 2]
    assert inc1.image_path is not None
    assert Path(inc1.image_path).exists()

    # Immediate second call should be suppressed by cooldown
    inc2 = mgr.record_incident(
        frame=dummy_frame,
        severity_level="MEDIUM",
        confidence_score=0.50,
        involved_track_ids=[3, 4],
        people_near=0,
        reasons=["Rear-end collision"],
        source_type="TestVideo"
    )

    assert inc2.incident_id == inc1.incident_id, "Cooldown should return active incident rather than creating duplicate."
