"""
NexGuard — Core Test Suite
============================
Tests that run without external services, real cameras, or WhatsApp credentials.
All external services are mocked.

Run with:
    python -m pytest tests/ -v
"""

import math
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# ── Allow running from project root ───────────────────────────────────────────
sys.path.insert(0, ".")


class TestConfig(unittest.TestCase):
    """Configuration system tests."""

    def test_config_loads(self):
        from config import cfg
        self.assertIsNotNone(cfg)
        self.assertIsInstance(cfg.confidence_threshold, float)
        self.assertIsInstance(cfg.model_path, str)

    def test_confidence_in_range(self):
        from config import cfg
        self.assertGreaterEqual(cfg.confidence_threshold, 0.0)
        self.assertLessEqual(cfg.confidence_threshold, 1.0)

    def test_accident_threshold_in_range(self):
        from config import cfg
        self.assertGreaterEqual(cfg.accident_threshold, 0.0)
        self.assertLessEqual(cfg.accident_threshold, 1.0)

    def test_vehicle_classes_not_empty(self):
        from config import cfg
        self.assertGreater(len(cfg.vehicle_classes), 0)
        self.assertIn("car", cfg.vehicle_classes)

    def test_all_target_classes(self):
        from config import cfg
        classes = cfg.all_target_classes
        self.assertIn("person", classes)
        self.assertIn("car", classes)

    def test_severity_thresholds_ordered(self):
        from config import cfg
        self.assertLess(cfg.severity_low, cfg.severity_medium)
        self.assertLess(cfg.severity_medium, cfg.severity_high)
        self.assertLess(cfg.severity_high, cfg.severity_critical)

    def test_get_contacts_for_severity_fallback(self):
        from config import cfg
        cfg.whatsapp_recipients = ["9591152862"]
        cfg.alert_high_contacts = []
        contacts = cfg.get_contacts_for_severity("HIGH")
        self.assertIn("9591152862", contacts)


class TestSchemas(unittest.TestCase):
    """Data model tests."""

    def test_detection_center(self):
        from nexguard.models.schemas import Detection
        d = Detection(bbox=(100.0, 100.0, 200.0, 200.0), class_name="car", confidence=0.9)
        cx, cy = d.center
        self.assertAlmostEqual(cx, 150.0)
        self.assertAlmostEqual(cy, 150.0)

    def test_detection_area(self):
        from nexguard.models.schemas import Detection
        d = Detection(bbox=(0.0, 0.0, 100.0, 100.0), class_name="car", confidence=0.9)
        self.assertAlmostEqual(d.area, 10000.0)

    def test_track_speed_drop_ratio_no_history(self):
        from nexguard.models.schemas import Track
        t = Track(track_id=1, bbox=(0,0,100,100), class_name="car", confidence=0.9)
        self.assertEqual(t.speed_drop_ratio, 0.0)

    def test_track_speed_drop_ratio(self):
        from nexguard.models.schemas import Track
        t = Track(
            track_id=1, bbox=(0,0,100,100), class_name="car", confidence=0.9,
            speed=2.0, speed_history=[10.0, 10.0, 10.0, 2.0]
        )
        self.assertGreater(t.speed_drop_ratio, 0.5)

    def test_incident_to_dict(self):
        from nexguard.models.schemas import Incident, EventType, Severity, IncidentStatus
        inc = Incident(
            event_type=EventType.VEHICLE_COLLISION,
            severity=Severity.HIGH,
            confidence=0.85,
        )
        d = inc.to_dict()
        self.assertIn("incident_id", d)
        self.assertIn("severity", d)
        self.assertEqual(d["severity"], "HIGH")

    def test_incident_resolve(self):
        from nexguard.models.schemas import Incident, IncidentStatus
        inc = Incident()
        inc.resolve()
        self.assertEqual(inc.status, IncidentStatus.RESOLVED)
        self.assertIsNotNone(inc.end_time)

    def test_severity_result_to_dict(self):
        from nexguard.models.schemas import SeverityResult, Severity
        sr = SeverityResult(severity=Severity.HIGH, score=0.82, reasons=["test"])
        d = sr.to_dict()
        self.assertEqual(d["severity"], "HIGH")
        self.assertIn("score", d)


class TestTracker(unittest.TestCase):
    """Multi-object tracker tests."""

    def setUp(self):
        from nexguard.tracking.tracker import ObjectTracker
        self.tracker = ObjectTracker(max_lost_frames=5, history_length=10)

    def test_creates_tracks_from_detections(self):
        from nexguard.models.schemas import Detection
        dets = [
            Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9),
            Detection(bbox=(300, 300, 400, 400), class_name="person", confidence=0.85),
        ]
        tracks = self.tracker.update(dets)
        self.assertEqual(len(tracks), 2)

    def test_track_ids_are_unique(self):
        from nexguard.models.schemas import Detection
        dets = [
            Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9),
            Detection(bbox=(300, 300, 400, 400), class_name="car", confidence=0.8),
        ]
        tracks = self.tracker.update(dets)
        ids = [t.track_id for t in tracks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_track_persistence_across_frames(self):
        from nexguard.models.schemas import Detection
        dets_f1 = [Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9)]
        tracks_f1 = self.tracker.update(dets_f1)
        id_f1 = tracks_f1[0].track_id

        dets_f2 = [Detection(bbox=(105, 105, 205, 205), class_name="car", confidence=0.91)]
        tracks_f2 = self.tracker.update(dets_f2)
        id_f2 = tracks_f2[0].track_id

        self.assertEqual(id_f1, id_f2)  # Same object, same ID

    def test_track_removed_after_lost_frames(self):
        from nexguard.models.schemas import Detection
        dets = [Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9)]
        self.tracker.update(dets)

        # Feed empty frames until track is removed
        for _ in range(10):
            self.tracker.update([])

        self.assertEqual(self.tracker.active_count, 0)

    def test_velocity_computed(self):
        from nexguard.models.schemas import Detection
        dets_f1 = [Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9)]
        self.tracker.update(dets_f1)

        dets_f2 = [Detection(bbox=(120, 100, 220, 200), class_name="car", confidence=0.9)]
        tracks_f2 = self.tracker.update(dets_f2)

        self.assertGreater(tracks_f2[0].speed, 0)

    def test_is_vehicle_flag(self):
        from nexguard.models.schemas import Detection
        dets = [Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9)]
        tracks = self.tracker.update(dets)
        self.assertTrue(tracks[0].is_vehicle)
        self.assertFalse(tracks[0].is_person)

    def test_get_vehicles_and_persons(self):
        from nexguard.models.schemas import Detection
        dets = [
            Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9),
            Detection(bbox=(300, 300, 360, 460), class_name="person", confidence=0.85),
        ]
        self.tracker.update(dets)
        self.assertEqual(len(self.tracker.get_vehicles()), 1)
        self.assertEqual(len(self.tracker.get_persons()), 1)

    def test_reset_clears_tracks(self):
        from nexguard.models.schemas import Detection
        dets = [Detection(bbox=(100, 100, 200, 200), class_name="car", confidence=0.9)]
        self.tracker.update(dets)
        self.tracker.reset()
        self.assertEqual(self.tracker.active_count, 0)


class TestAccidentDetector(unittest.TestCase):
    """Accident detection engine tests."""

    def setUp(self):
        from nexguard.accident.accident_detector import AccidentDetector
        self.detector = AccidentDetector(window=20, threshold=0.55, confirm_frames=3)

    def _make_track(self, tid, cx, cy, speed=0.0, is_vehicle=True, speed_history=None):
        from nexguard.models.schemas import Track
        w, h = 60, 40
        t = Track(
            track_id=tid,
            bbox=(cx - w/2, cy - h/2, cx + w/2, cy + h/2),
            class_name="car" if is_vehicle else "person",
            confidence=0.9,
            cx=cx, cy=cy,
            speed=speed,
            is_vehicle=is_vehicle,
            is_person=not is_vehicle,
            age=10,
            speed_history=speed_history or [speed] * 5,
        )
        return t

    def test_no_accident_with_distant_tracks(self):
        """Well-separated, slow-moving vehicles should not trigger accidents."""
        for _ in range(10):
            tracks = [
                self._make_track(1, cx=100, cy=100, speed=2.0),
                self._make_track(2, cx=800, cy=500, speed=2.0),
            ]
            event = self.detector.update(tracks)
            self.assertIsNone(event)

    def test_collision_score_increases_with_overlap(self):
        """Heavily overlapping vehicles should score higher than separated ones."""
        from nexguard.accident.accident_detector import _iou
        b1 = (100, 100, 200, 200)
        b2 = (150, 150, 250, 250)  # heavy overlap
        b3 = (500, 500, 600, 600)  # no overlap

        iou_overlap = _iou(b1, b2)
        iou_none = _iou(b1, b3)

        self.assertGreater(iou_overlap, 0.0)
        self.assertAlmostEqual(iou_none, 0.0)

    def test_reset_clears_state(self):
        tracks = [self._make_track(1, cx=200, cy=200, speed=5.0)]
        for _ in range(5):
            self.detector.update(tracks)
        self.detector.reset()
        self.assertEqual(self.detector.consecutive_count, 0)


class TestSeverityClassifier(unittest.TestCase):
    """Severity classification tests."""

    def setUp(self):
        from nexguard.accident.severity import SeverityClassifier
        self.clf = SeverityClassifier()

    def _make_event(self, confidence=0.85):
        from nexguard.models.schemas import AccidentEvent, EventType
        from datetime import datetime
        return AccidentEvent(
            accident_detected=True,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            track_ids=[1, 2],
            location={"cx": 300.0, "cy": 200.0},
            event_type=EventType.VEHICLE_COLLISION,
            signals=["test signal"],
        )

    def _make_track(self, tid, is_vehicle=True):
        from nexguard.models.schemas import Track
        return Track(
            track_id=tid,
            bbox=(100, 100, 200, 200),
            class_name="car" if is_vehicle else "person",
            confidence=0.9,
            cx=150, cy=150,
            speed=2.0,
            speed_history=[8.0, 6.0, 4.0, 2.0],
            is_vehicle=is_vehicle,
            is_person=not is_vehicle,
        )

    def test_high_confidence_event_produces_nonzero_score(self):
        """High-confidence events must produce a meaningful severity score."""
        event = self._make_event(confidence=0.90)
        tracks = [self._make_track(1), self._make_track(2)]
        result = self.clf.classify(
            event, tracks, tracks,
            persistence_frames=20,  # long-lasting event → higher severity
        )
        # With high confidence + 2 vehicles + persistence, score must be > LOW threshold
        self.assertGreater(result.score, 0.30)

    def test_score_is_normalized(self):
        event = self._make_event(confidence=0.85)
        tracks = [self._make_track(1), self._make_track(2)]
        result = self.clf.classify(event, tracks, tracks)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_reasons_are_list(self):
        event = self._make_event(confidence=0.85)
        tracks = [self._make_track(1)]
        result = self.clf.classify(event, tracks, tracks)
        self.assertIsInstance(result.reasons, list)

    def test_severity_levels_ordered_by_score(self):
        from nexguard.models.schemas import Severity
        from config import cfg
        clf = self.clf
        self.assertEqual(clf._score_to_severity(0.0), Severity.LOW)
        self.assertEqual(clf._score_to_severity(cfg.severity_critical + 0.01), Severity.CRITICAL)


class TestPersonInvolvement(unittest.TestCase):
    """Person involvement analysis tests."""

    def _make_person_track(self, tid, cx, cy, speed=0.5):
        from nexguard.models.schemas import Track
        return Track(
            track_id=tid,
            bbox=(cx-20, cy-40, cx+20, cy+40),
            class_name="person",
            confidence=0.88,
            cx=cx, cy=cy,
            speed=speed,
            is_person=True,
            is_vehicle=False,
            speed_history=[speed] * 5,
        )

    def _make_event(self, cx, cy):
        from nexguard.models.schemas import AccidentEvent, EventType
        from datetime import datetime
        return AccidentEvent(
            accident_detected=True,
            confidence=0.85,
            timestamp=datetime.now().isoformat(),
            track_ids=[1, 2],
            location={"cx": cx, "cy": cy},
            event_type=EventType.VEHICLE_PEDESTRIAN,
            signals=[],
        )

    def test_person_close_to_event_is_high_risk(self):
        from nexguard.accident.severity import analyze_person_involvement
        person = self._make_person_track(10, cx=310, cy=210, speed=0.1)
        event = self._make_event(cx=300, cy=200)
        involvements = analyze_person_involvement([person], event, collision_radius_px=200)
        self.assertEqual(len(involvements), 1)
        self.assertEqual(involvements[0].involvement_level, "HIGH")

    def test_person_far_from_event_excluded(self):
        from nexguard.accident.severity import analyze_person_involvement
        person = self._make_person_track(11, cx=900, cy=900, speed=2.0)
        event = self._make_event(cx=300, cy=200)
        involvements = analyze_person_involvement([person], event, collision_radius_px=200)
        self.assertEqual(len(involvements), 0)


class TestIncidentManager(unittest.TestCase):
    """Incident lifecycle tests."""

    def _make_event(self, confidence=0.85):
        from nexguard.models.schemas import AccidentEvent, EventType
        from datetime import datetime
        return AccidentEvent(
            accident_detected=True,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            track_ids=[1, 2],
            location={"cx": 300.0, "cy": 200.0},
            event_type=EventType.VEHICLE_COLLISION,
            signals=["test"],
        )

    def _make_severity(self):
        from nexguard.models.schemas import SeverityResult, Severity
        return SeverityResult(severity=Severity.HIGH, score=0.80, reasons=["test"])

    def _make_tracks(self):
        from nexguard.models.schemas import Track
        return [
            Track(track_id=1, bbox=(100,100,200,200), class_name="car",
                  confidence=0.9, is_vehicle=True),
            Track(track_id=2, bbox=(150,100,250,200), class_name="car",
                  confidence=0.88, is_vehicle=True),
        ]

    def test_creates_incident(self):
        from nexguard.accident.event import IncidentManager
        manager = IncidentManager()
        event = self._make_event()
        severity = self._make_severity()
        tracks = self._make_tracks()
        incident = manager.create_incident(event, severity, tracks)
        self.assertIsNotNone(incident.incident_id)
        self.assertEqual(manager.total_incidents, 1)

    def test_first_alert_always_allowed(self):
        from nexguard.accident.event import IncidentManager
        from nexguard.models.schemas import IncidentStatus
        manager = IncidentManager()
        event = self._make_event()
        severity = self._make_severity()
        tracks = self._make_tracks()
        incident = manager.create_incident(event, severity, tracks)
        self.assertTrue(manager.should_send_alert(incident))

    def test_alert_blocked_after_cooldown_not_expired(self):
        from nexguard.accident.event import IncidentManager
        from nexguard.models.schemas import IncidentStatus, Severity
        manager = IncidentManager(alert_cooldown=3600)  # 1 hour
        event = self._make_event()
        severity = self._make_severity()
        tracks = self._make_tracks()
        incident = manager.create_incident(event, severity, tracks)
        manager.mark_alert_sent(incident)
        # Should be blocked until cooldown expires
        self.assertFalse(manager.should_send_alert(incident))

    def test_incident_resolves_when_tracks_gone(self):
        from nexguard.accident.event import IncidentManager
        manager = IncidentManager()
        event = self._make_event()
        severity = self._make_severity()
        tracks = self._make_tracks()
        incident = manager.create_incident(event, severity, tracks)
        # Feed active track IDs that don't include the incident tracks
        resolved = manager.resolve_stale_incidents(active_track_ids=[99, 100])
        self.assertEqual(len(resolved), 1)
        self.assertEqual(manager.total_incidents, 1)


class TestAlertManagerMocked(unittest.TestCase):
    """Alert manager tests with mocked WhatsApp."""

    def test_dispatch_calls_notifier(self):
        from nexguard.alerts.manager import AlertManager
        from nexguard.models.schemas import Incident, Severity, IncidentStatus, EventType

        mock_notifier = MagicMock()
        manager = AlertManager(notifier=mock_notifier)

        from config import cfg
        cfg.whatsapp_recipients = ["9591152862"]
        cfg.alert_high_contacts = []
        cfg.alert_medium_contacts = []
        cfg.alert_critical_contacts = []

        incident = Incident(
            severity=Severity.HIGH,
            severity_score=0.82,
            confidence=0.88,
            event_type=EventType.VEHICLE_COLLISION,
            involved_vehicles=[1, 2],
        )
        manager.dispatch(incident)
        mock_notifier.send.assert_called_once()

    def test_low_severity_no_send_without_contacts(self):
        from nexguard.alerts.manager import AlertManager
        from nexguard.models.schemas import Incident, Severity, EventType
        from config import cfg

        mock_notifier = MagicMock()
        manager = AlertManager(notifier=mock_notifier)

        cfg.alert_low_contacts = []

        incident = Incident(
            severity=Severity.LOW,
            severity_score=0.25,
            confidence=0.62,
            event_type=EventType.NEAR_MISS,
        )
        result = manager.dispatch(incident)
        self.assertFalse(result)
        mock_notifier.send.assert_not_called()


class TestVideoSource(unittest.TestCase):
    """Video source type detection tests."""

    def test_webcam_detection(self):
        from nexguard.input.source import VideoSource
        vs = VideoSource("0")
        self.assertEqual(vs.source_type, "webcam")

    def test_file_detection(self):
        from nexguard.input.source import VideoSource
        vs = VideoSource("test_clips/road.mp4")
        self.assertEqual(vs.source_type, "file")

    def test_rtsp_detection(self):
        from nexguard.input.source import VideoSource
        vs = VideoSource("rtsp://192.168.1.1:554/stream")
        self.assertEqual(vs.source_type, "rtsp")

    def test_missing_file_raises(self):
        from nexguard.input.source import VideoSource
        vs = VideoSource("/nonexistent/path/video.mp4")
        with self.assertRaises(FileNotFoundError):
            vs.open()


if __name__ == "__main__":
    unittest.main(verbosity=2)
