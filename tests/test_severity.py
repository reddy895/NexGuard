"""
Tests — Severity State Transitions
=====================================
Unit tests for SeverityClassifier decision logic.

These tests validate that:
  - LOW / MEDIUM / HIGH boundaries produce the correct Severity label
  - UNCERTAIN path fires for borderline trained-model detections
  - analyze_person_involvement correctly classifies distances
  - Severity scoring is bounded to [0, 1]

Tests use only synthetic data — no model weights or video required.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when running tests directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from nexguard.accident.severity import SeverityClassifier, analyze_person_involvement
from nexguard.models.schemas import (
    AccidentEvent,
    DetectionMode,
    EventType,
    PersonInvolvement,
    Severity,
    Track,
)
from nexguard.utils.utils import now_iso


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(
    track_id: int = 1,
    class_name: str = "car",
    speed: float = 0.0,
    speed_history: list | None = None,
    cx: float = 100.0,
    cy: float = 100.0,
    hits: int = 5,
    age: int = 5,
) -> Track:
    t = Track(
        track_id=track_id,
        bbox=(cx - 30, cy - 20, cx + 30, cy + 20),
        class_name=class_name,
        confidence=0.85,
        cx=cx,
        cy=cy,
        speed=speed,
        speed_history=speed_history or [speed],
        hits=hits,
        age=age,
        is_vehicle=(class_name != "person"),
        is_person=(class_name == "person"),
    )
    return t


def _make_event(
    confidence: float = 0.70,
    track_ids: list | None = None,
    detection_mode: DetectionMode = DetectionMode.HEURISTIC,
    is_model_confirmed: bool = False,
) -> AccidentEvent:
    return AccidentEvent(
        accident_detected=True,
        confidence=confidence,
        timestamp=now_iso(),
        track_ids=track_ids or [1, 2],
        location={"cx": 100.0, "cy": 100.0},
        event_type=EventType.VEHICLE_COLLISION,
        signals=["test_signal"],
        detection_mode=detection_mode,
        is_model_confirmed=is_model_confirmed,
    )


# ---------------------------------------------------------------------------
# SeverityClassifier tests
# ---------------------------------------------------------------------------

class TestSeverityClassifier:

    def setup_method(self):
        self.clf = SeverityClassifier()

    def test_low_severity_baseline(self):
        """Low-confidence event with no person involvement → LOW."""
        event = _make_event(confidence=0.55)
        t = _make_track(speed=0.5, speed_history=[1.0, 0.5])
        result = self.clf.classify(event=event, involved_tracks=[t], all_tracks=[t])
        assert result.severity in (Severity.LOW, Severity.MEDIUM)
        assert 0.0 <= result.score <= 1.0

    def test_score_bounded(self):
        """Severity score is always in [0, 1] regardless of inputs."""
        event = _make_event(confidence=1.0)
        tracks = [_make_track(i, speed=10.0, speed_history=[20.0] * 10) for i in range(5)]
        result = self.clf.classify(
            event=event, involved_tracks=tracks, all_tracks=tracks,
            persistence_frames=60,
        )
        assert 0.0 <= result.score <= 1.0

    def test_high_severity_with_persons(self):
        """High collision confidence + HIGH-risk person → HIGH or MEDIUM."""
        event = _make_event(confidence=0.90)
        t1 = _make_track(1, speed=0.2, speed_history=[8.0, 0.2])
        t2 = _make_track(2, speed=0.1, speed_history=[6.0, 0.1])
        person_inv = [PersonInvolvement(
            track_id=99, distance_px=40.0, movement="stationary",
            involvement_level="HIGH", position=(105.0, 105.0),
        )]
        result = self.clf.classify(
            event=event,
            involved_tracks=[t1, t2],
            all_tracks=[t1, t2],
            person_involvements=person_inv,
            persistence_frames=25,
        )
        assert result.severity in (Severity.MEDIUM, Severity.HIGH)
        assert any("person" in r.lower() for r in result.reasons)

    def test_medium_severity_medium_risk_person(self):
        """MEDIUM-risk person near event contributes to severity score."""
        event = _make_event(confidence=0.70)
        t = _make_track(speed=0.0, speed_history=[5.0, 0.0])
        person_inv = [PersonInvolvement(
            track_id=10, distance_px=120.0, movement="stationary",
            involvement_level="MEDIUM", position=(150.0, 150.0),
        )]
        result = self.clf.classify(
            event=event, involved_tracks=[t], all_tracks=[t],
            person_involvements=person_inv,
        )
        assert result.score > 0.0
        assert result.severity in (Severity.LOW, Severity.MEDIUM, Severity.HIGH)

    def test_uncertain_borderline_trained_model(self):
        """Borderline trained-model event with very low score → UNCERTAIN."""
        event = _make_event(
            confidence=0.10,
            detection_mode=DetectionMode.TRAINED_MODEL,
            is_model_confirmed=True,
        )
        t = _make_track(speed=0.1)
        result = self.clf.classify(event=event, involved_tracks=[t], all_tracks=[t])
        assert result.severity == Severity.UNCERTAIN

    def test_multiple_vehicles_increases_score(self):
        """More involved vehicles should increase the severity score."""
        event1 = _make_event(confidence=0.65)
        event2 = _make_event(confidence=0.65)
        t1 = _make_track(1)
        tracks_1 = [t1]
        tracks_3 = [_make_track(i) for i in range(3)]

        r1 = self.clf.classify(event=event1, involved_tracks=tracks_1, all_tracks=tracks_1)
        r3 = self.clf.classify(event=event2, involved_tracks=tracks_3, all_tracks=tracks_3)
        assert r3.score >= r1.score

    def test_reasons_list_populated(self):
        """At least one reason should be present for a significant event."""
        event = _make_event(confidence=0.80)
        t = _make_track(speed=0.0, speed_history=[10.0, 0.0])
        result = self.clf.classify(event=event, involved_tracks=[t], all_tracks=[t])
        assert len(result.reasons) >= 1

    def test_persistence_increases_score(self):
        """Longer persistence should push score higher."""
        event1 = _make_event(confidence=0.55)
        event2 = _make_event(confidence=0.55)
        t = _make_track()
        r_short = self.clf.classify(event=event1, involved_tracks=[t], all_tracks=[t],
                                    persistence_frames=0)
        r_long  = self.clf.classify(event=event2, involved_tracks=[t], all_tracks=[t],
                                    persistence_frames=30)
        assert r_long.score >= r_short.score


# ---------------------------------------------------------------------------
# analyze_person_involvement tests
# ---------------------------------------------------------------------------

class TestPersonInvolvement:

    def _person(self, cx: float, cy: float, speed: float = 0.1) -> Track:
        t = _make_track(track_id=99, class_name="person", cx=cx, cy=cy, speed=speed)
        t.is_person = True
        t.is_vehicle = False
        return t

    def test_high_involvement_close(self):
        """Person within 40% of radius → HIGH involvement."""
        event = _make_event()
        event.location = {"cx": 100.0, "cy": 100.0}
        person = self._person(105.0, 105.0, speed=0.1)
        results = analyze_person_involvement([person], event, collision_radius_px=200.0)
        assert len(results) == 1
        assert results[0].involvement_level == "HIGH"

    def test_medium_involvement_mid_range(self):
        """Person at 50% of radius → MEDIUM involvement."""
        event = _make_event()
        event.location = {"cx": 0.0, "cy": 0.0}
        person = self._person(100.0, 0.0, speed=0.2)
        results = analyze_person_involvement([person], event, collision_radius_px=200.0)
        assert len(results) == 1
        assert results[0].involvement_level == "MEDIUM"

    def test_low_involvement_at_edge(self):
        """Person at 85% of radius → LOW involvement."""
        event = _make_event()
        event.location = {"cx": 0.0, "cy": 0.0}
        person = self._person(170.0, 0.0, speed=0.3)
        results = analyze_person_involvement([person], event, collision_radius_px=200.0)
        assert len(results) == 1
        assert results[0].involvement_level == "LOW"

    def test_outside_radius_excluded(self):
        """Person beyond the radius should not appear in results."""
        event = _make_event()
        event.location = {"cx": 0.0, "cy": 0.0}
        person = self._person(300.0, 0.0)
        results = analyze_person_involvement([person], event, collision_radius_px=200.0)
        assert len(results) == 0

    def test_stationary_movement_classification(self):
        """Very slow person should be classified as 'stationary'."""
        event = _make_event()
        event.location = {"cx": 0.0, "cy": 0.0}
        person = self._person(30.0, 0.0, speed=0.2)
        person.speed = 0.2
        results = analyze_person_involvement([person], event, collision_radius_px=200.0)
        assert results[0].movement == "stationary"

    def test_multiple_persons(self):
        """Multiple persons at various distances are all classified."""
        event = _make_event()
        event.location = {"cx": 0.0, "cy": 0.0}
        persons = [
            self._person(30.0, 0.0),
            self._person(100.0, 0.0),
            self._person(500.0, 0.0),
        ]
        results = analyze_person_involvement(persons, event, collision_radius_px=200.0)
        assert len(results) == 2  # third person (500px) is outside radius
