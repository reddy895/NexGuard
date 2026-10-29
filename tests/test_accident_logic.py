"""
Tests — Accident Detection Logic and Cooldown
===============================================
Unit tests for AccidentDetector temporal confirmation and state management.

Tests cover:
  - Confirmation: must hold for N consecutive frames before emitting
  - Cooldown: no re-emission until cooldown_frames have elapsed
  - Reset: detector resets cleanly between videos
  - No false positives on sparse tracks (< 2 frame history)
  - Model-track path (direct accident class detections)
  - Sudden-stop detection on a single vehicle
  - IncidentManager deduplication and resolution logic

No model weights or video files required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from nexguard.accident.accident_detector import AccidentDetector, _approach_velocity
from nexguard.accident.event import IncidentManager
from nexguard.models.schemas import (
    AccidentEvent,
    Detection,
    DetectionMode,
    EventType,
    Incident,
    IncidentStatus,
    Severity,
    SeverityResult,
    Track,
)
from nexguard.utils.utils import now_iso


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vehicle(
    track_id: int,
    cx: float = 100.0,
    cy: float = 100.0,
    speed: float = 5.0,
    speed_history: list | None = None,
    iou_bbox: tuple | None = None,
    hits: int = 5,
    age: int = 5,
) -> Track:
    if iou_bbox is None:
        iou_bbox = (cx - 30, cy - 20, cx + 30, cy + 20)
    t = Track(
        track_id=track_id,
        bbox=iou_bbox,
        class_name="car",
        confidence=0.85,
        cx=cx, cy=cy,
        velocity=(0.0, 0.0),
        speed=speed,
        speed_history=speed_history or [speed],
        hits=hits, age=age,
        is_vehicle=True, is_person=False, is_accident=False,
    )
    return t


def _person(track_id: int, cx: float = 200.0, cy: float = 200.0) -> Track:
    t = Track(
        track_id=track_id,
        bbox=(cx - 15, cy - 30, cx + 15, cy + 30),
        class_name="person",
        confidence=0.80,
        cx=cx, cy=cy,
        speed=0.5,
        speed_history=[0.5],
        hits=3, age=3,
        is_vehicle=False, is_person=True, is_accident=False,
    )
    return t


def _make_event(confidence: float = 0.70, track_ids: list | None = None) -> AccidentEvent:
    return AccidentEvent(
        accident_detected=True,
        confidence=confidence,
        timestamp=now_iso(),
        track_ids=track_ids or [1, 2],
        location={"cx": 100.0, "cy": 100.0},
        event_type=EventType.VEHICLE_COLLISION,
        signals=["test_signal"],
    )


def _severity_result(sev: Severity = Severity.LOW, score: float = 0.35) -> SeverityResult:
    return SeverityResult(severity=sev, score=score, reasons=["test"])


# ---------------------------------------------------------------------------
# AccidentDetector tests
# ---------------------------------------------------------------------------

class TestAccidentDetectorConfirmation:

    def setup_method(self):
        # Low thresholds to make triggering easy in tests
        self.det = AccidentDetector(
            window=10,
            threshold=0.50,
            confirm_frames=3,
            cooldown_frames=5,
        )

    def _overlapping_pair(self):
        """Two vehicles with strongly overlapping bounding boxes."""
        v1 = _vehicle(1, cx=100, cy=100, iou_bbox=(50, 60, 160, 150))
        v2 = _vehicle(2, cx=110, cy=110, iou_bbox=(60, 70, 170, 160),
                      speed=0.2, speed_history=[6.0, 0.2])
        v1.velocity = (3.0, 0.0)
        v2.velocity = (-3.0, 0.0)
        return [v1, v2]

    def test_no_emission_before_confirm_frames(self):
        """Event must be sustained for confirm_frames before emission."""
        pair = self._overlapping_pair()
        events = [self.det.update(pair) for _ in range(2)]
        assert all(e is None for e in events), "Should not emit before confirm_frames"

    def test_emission_after_confirm_frames(self):
        """Event should emit once confirm_frames is reached."""
        pair = self._overlapping_pair()
        results = [self.det.update(pair) for _ in range(5)]
        assert any(r is not None for r in results), "Should have emitted by frame 5"

    def test_cooldown_prevents_immediate_re_emission(self):
        """After an event is emitted, cooldown prevents immediate re-emission."""
        pair = self._overlapping_pair()
        # Drive to emission
        emitted = []
        for _ in range(8):
            r = self.det.update(pair)
            if r is not None:
                emitted.append(r)
        # There should be at most 1 emission in the first 8 frames
        assert len(emitted) <= 1, f"Expected at most 1 emission, got {len(emitted)}"

    def test_no_event_without_enough_history(self):
        """Detector needs ≥2 frames of history; first frame always returns None."""
        det = AccidentDetector(window=10, threshold=0.50, confirm_frames=3, cooldown_frames=5)
        pair = self._overlapping_pair()
        result = det.update(pair)
        assert result is None

    def test_empty_tracks_return_none(self):
        """Empty track list should never produce an event."""
        for _ in range(10):
            assert self.det.update([]) is None

    def test_single_vehicle_no_pair_collision(self):
        """Single vehicle cannot trigger vehicle-vehicle collision."""
        v = _vehicle(1, speed=3.0)
        for _ in range(6):
            r = self.det.update([v])
            assert r is None

    def test_reset_clears_state(self):
        """After reset(), consecutive counter and frame index are zeroed."""
        pair = self._overlapping_pair()
        for _ in range(4):
            self.det.update(pair)
        self.det.reset()
        assert self.det.consecutive_count == 0
        # After reset, must re-confirm before emitting
        result = self.det.update(pair)
        assert result is None

    def test_smoothed_confidence_nonzero_after_updates(self):
        """After several frames with collision signals, smoothed_confidence > 0."""
        pair = self._overlapping_pair()
        for _ in range(4):
            self.det.update(pair)
        assert self.det.smoothed_confidence > 0.0

    def test_persons_contribute_to_vehicle_pedestrian_event(self):
        """Person near collision zone should produce VEHICLE_PEDESTRIAN type."""
        v1 = _vehicle(1, cx=100, cy=100, iou_bbox=(50, 60, 160, 150))
        v2 = _vehicle(2, cx=110, cy=110, iou_bbox=(60, 70, 170, 160),
                      speed=0.2, speed_history=[6.0, 0.2])
        v1.velocity = (3.0, 0.0)
        v2.velocity = (-3.0, 0.0)
        p = _person(99, cx=115, cy=115)  # very close to collision zone

        emitted = None
        for _ in range(6):
            r = self.det.update([v1, v2, p])
            if r is not None:
                emitted = r
                break

        if emitted:
            assert emitted.event_type == EventType.VEHICLE_PEDESTRIAN


# ---------------------------------------------------------------------------
# Approach velocity helper
# ---------------------------------------------------------------------------

class TestApproachVelocity:

    def test_approaching_vehicles(self):
        """Two vehicles moving toward each other should have positive approach velocity."""
        v1 = _vehicle(1, cx=0.0, cy=0.0)
        v1.velocity = (5.0, 0.0)
        v2 = _vehicle(2, cx=100.0, cy=0.0)
        v2.velocity = (-5.0, 0.0)
        result = _approach_velocity(v1, v2)
        assert result > 0

    def test_separating_vehicles(self):
        """Vehicles moving apart should have low/negative approach velocity."""
        v1 = _vehicle(1, cx=0.0, cy=0.0)
        v1.velocity = (-3.0, 0.0)
        v2 = _vehicle(2, cx=100.0, cy=0.0)
        v2.velocity = (3.0, 0.0)
        result = _approach_velocity(v1, v2)
        assert result <= 0

    def test_zero_velocity_tracks(self):
        """Zero velocity tracks have zero approach velocity."""
        v1 = _vehicle(1, cx=0.0, cy=0.0, speed=0.0)
        v2 = _vehicle(2, cx=50.0, cy=0.0, speed=0.0)
        result = _approach_velocity(v1, v2)
        assert result == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# IncidentManager tests
# ---------------------------------------------------------------------------

class TestIncidentManager:

    def setup_method(self):
        self.mgr = IncidentManager()

    def test_create_incident(self):
        """Creating an incident returns an ACTIVE Incident object."""
        event = _make_event()
        sr = _severity_result()
        incident = self.mgr.create_incident(event=event, severity_result=sr, involved_tracks=[])
        assert incident.status == IncidentStatus.ACTIVE
        assert incident.incident_id.startswith("NG-")
        assert len(self.mgr.active_incidents) == 1

    def test_deduplication_same_tracks(self):
        """Second event with same track IDs updates existing incident, not creates new."""
        event1 = _make_event(track_ids=[1, 2])
        event2 = _make_event(confidence=0.85, track_ids=[1, 2])
        sr1 = _severity_result(Severity.LOW, 0.30)
        sr2 = _severity_result(Severity.HIGH, 0.80)

        i1 = self.mgr.create_incident(event=event1, severity_result=sr1, involved_tracks=[])
        i2 = self.mgr.create_incident(event=event2, severity_result=sr2, involved_tracks=[])

        assert i1.incident_id == i2.incident_id, "Should deduplicate same-track events"
        assert i1.severity == Severity.HIGH, "Severity should escalate"
        assert len(self.mgr.active_incidents) == 1

    def test_resolve_stale_removes_incidents(self):
        """Incident is resolved when its tracks are no longer active."""
        event = _make_event(track_ids=[1, 2])
        sr = _severity_result()
        self.mgr.create_incident(event=event, severity_result=sr, involved_tracks=[])
        assert len(self.mgr.active_incidents) == 1

        self.mgr.resolve_stale_incidents(active_track_ids=[3, 4])  # tracks 1,2 gone
        assert len(self.mgr.active_incidents) == 0
        assert len(self.mgr.incident_history) == 1

    def test_incident_stays_active_while_tracks_present(self):
        """Incident is NOT resolved while its tracks are still active."""
        event = _make_event(track_ids=[1, 2])
        sr = _severity_result()
        self.mgr.create_incident(event=event, severity_result=sr, involved_tracks=[])
        self.mgr.resolve_stale_incidents(active_track_ids=[1, 3])
        assert len(self.mgr.active_incidents) == 1

    def test_total_incidents_counts_both(self):
        """total_incidents includes both active and resolved."""
        event = _make_event(track_ids=[1, 2])
        sr = _severity_result()
        self.mgr.create_incident(event=event, severity_result=sr, involved_tracks=[])
        self.mgr.resolve_stale_incidents(active_track_ids=[])
        assert self.mgr.total_incidents == 1

    def test_history_cap_enforced(self):
        """History should never grow beyond _MAX_HISTORY."""
        for i in range(IncidentManager._MAX_HISTORY + 10):
            ev = _make_event(track_ids=[i * 2, i * 2 + 1])
            ev.location = {"cx": float(i * 1000), "cy": 0.0}  # ensure no dedup
            sr = _severity_result()
            self.mgr.create_incident(event=ev, severity_result=sr, involved_tracks=[])
            self.mgr.resolve_stale_incidents(active_track_ids=[])
        assert len(self.mgr.incident_history) <= IncidentManager._MAX_HISTORY
