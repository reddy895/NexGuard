"""
Unit Tests for NexGuard Evidence Collector
===========================================
"""

from pathlib import Path
import numpy as np
import pytest
from nexguard.evidence.collector import EvidenceCollector
from nexguard.models.schemas import Incident, Severity, EventType

def test_evidence_collector_saves_snapshot_and_metadata(tmp_path: Path):
    collector = EvidenceCollector(evidence_dir=tmp_path / "evidence")

    inc = Incident(
        incident_id="NG-TEST1",
        severity=Severity.HIGH,
        confidence=0.88,
        event_type=EventType.VEHICLE_COLLISION,
        involved_tracks=[1, 2],
        severity_reasons=["high impact speed drop", "multiple vehicles"],
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    record = collector.capture_evidence(inc, dummy_frame, frame_index=42)

    assert Path(record.snapshot_path).exists()
    assert Path(record.metadata_path).exists()
    assert record.incident_id == "NG-TEST1"
    assert record.frame_index == 42
    assert record.severity == "HIGH"
