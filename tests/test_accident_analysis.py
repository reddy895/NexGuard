import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.accident_detector import AccidentDetector, build_detection_record
from src.severity_analyzer import SeverityAnalyzer


class TestAccidentAnalysis(unittest.TestCase):
    def test_person_detection_data_is_correctly_represented(self):
        detection = build_detection_record(
            track_id=12,
            class_id=0,
            class_name="person",
            confidence=0.91,
            box=[10, 20, 50, 100],
        )

        self.assertEqual(detection["track_id"], 12)
        self.assertEqual(detection["class_name"], "person")
        self.assertAlmostEqual(detection["confidence"], 0.91)
        self.assertEqual(detection["bbox"], [10, 20, 50, 100])
        self.assertEqual(detection["center"], [30.0, 60.0])

    def test_vehicle_detection_data_is_correctly_represented(self):
        detection = build_detection_record(
            track_id=7,
            class_id=2,
            class_name="car",
            confidence=0.88,
            box=[120, 80, 280, 220],
        )

        self.assertEqual(detection["track_id"], 7)
        self.assertEqual(detection["class_name"], "car")
        self.assertEqual(detection["bbox"], [120, 80, 280, 220])
        self.assertEqual(detection["center"], [200.0, 150.0])

    def test_tracking_history_is_bounded(self):
        tracker = AccidentDetector(max_history=3)
        for idx in range(6):
            tracker.update_tracks([
                {
                    "track_id": 1,
                    "class_name": "car",
                    "confidence": 0.8,
                    "bbox": [100 + idx, 100, 200 + idx, 200],
                    "center": [150 + idx, 150],
                    "velocity": [1.0, 0.0],
                }
            ])

        self.assertLessEqual(len(tracker.tracked_objects[1]["history"]), 3)

    def test_normal_vehicle_movement_does_not_immediately_trigger_an_accident(self):
        tracker = AccidentDetector(candidate_threshold=0.65, confirmation_frames=3)
        normal_objects = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.84,
                "bbox": [100, 100, 200, 200],
                "center": [150, 150],
                "velocity": [7.0, 0.0],
                "history": [[150, 150], [155, 150]],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.86,
                "bbox": [300, 100, 400, 200],
                "center": [350, 150],
                "velocity": [8.0, 0.0],
                "history": [[350, 150], [355, 150]],
            },
        ]

        event = tracker.analyze_frame(normal_objects, frame_shape=(480, 640))
        self.assertIsNone(event)

    def test_synthetic_collision_like_sequence_can_trigger_an_accident_candidate(self):
        tracker = AccidentDetector(candidate_threshold=0.4, confirmation_frames=3)
        collision_objects = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.88,
                "bbox": [180, 150, 260, 230],
                "center": [220, 190],
                "velocity": [18.0, 2.0],
                "history": [[220, 190], [210, 190]],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.89,
                "bbox": [340, 150, 420, 230],
                "center": [380, 190],
                "velocity": [-18.0, -1.0],
                "history": [[380, 190], [390, 190]],
            },
        ]

        event = tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        self.assertIsNotNone(event)
        self.assertIn(event["status"], {"suspected", "confirmed"})

    def test_accident_confirmation_requires_multiple_frames(self):
        tracker = AccidentDetector(candidate_threshold=0.38, confirmation_frames=3, cooldown_frames=10)
        collision_objects = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [180, 150, 260, 230],
                "center": [220, 190],
                "velocity": [14.0, 2.0],
                "history": [[220, 190], [205, 188]],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [340, 150, 420, 230],
                "center": [380, 190],
                "velocity": [-14.0, -2.0],
                "history": [[380, 190], [395, 192]],
            },
        ]

        self.assertIsNotNone(tracker.analyze_frame(collision_objects, frame_shape=(480, 640)))
        self.assertIsNotNone(tracker.analyze_frame(collision_objects, frame_shape=(480, 640)))
        confirmed = tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        self.assertIsNotNone(confirmed)
        self.assertEqual(confirmed["status"], "confirmed")

    def test_confirmed_accident_creates_an_event_object(self):
        tracker = AccidentDetector(candidate_threshold=0.38, confirmation_frames=2, cooldown_frames=10)
        collision_objects = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.91,
                "bbox": [180, 140, 260, 230],
                "center": [220, 185],
                "velocity": [19.0, 5.0],
                "history": [[220, 185], [205, 180]],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.92,
                "bbox": [340, 140, 420, 230],
                "center": [380, 185],
                "velocity": [-19.0, -5.0],
                "history": [[380, 185], [395, 190]],
            },
        ]

        tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        confirmed = tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        self.assertIsNotNone(confirmed)
        self.assertEqual(confirmed["event_type"], "accident")
        self.assertIn(confirmed["severity"], {"LOW", "MEDIUM", "HIGH"})
        self.assertGreaterEqual(confirmed["confidence"], 0.0)

    def test_cooldown_prevents_duplicate_accident_events(self):
        tracker = AccidentDetector(candidate_threshold=0.38, confirmation_frames=2, cooldown_frames=5)
        collision_objects = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.91,
                "bbox": [180, 140, 260, 230],
                "center": [220, 185],
                "velocity": [22.0, 5.0],
                "history": [[220, 185], [205, 180]],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.92,
                "bbox": [340, 140, 420, 230],
                "center": [380, 185],
                "velocity": [-22.0, -5.0],
                "history": [[380, 185], [395, 190]],
            },
        ]

        tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        first_event = tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        self.assertIsNotNone(first_event)
        duplicate = tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
        self.assertIsNone(duplicate)

    def test_incident_evidence_is_saved_correctly(self):
        tracker = AccidentDetector(candidate_threshold=0.38, confirmation_frames=2, cooldown_frames=3)
        collision_objects = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [150, 100, 250, 200],
                "center": [200, 150],
                "velocity": [18.0, 4.0],
                "history": [[200, 150], [185, 145]],
            },
            {
                "track_id": 2,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": [350, 100, 450, 200],
                "center": [400, 150],
                "velocity": [-18.0, -4.0],
                "history": [[400, 150], [415, 155]],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker.output_dir = tmp_dir
            tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
            confirmed = tracker.analyze_frame(collision_objects, frame_shape=(480, 640))
            self.assertIsNotNone(confirmed)
            saved_path = Path(confirmed["evidence_path"])
            self.assertTrue(saved_path.exists())
            self.assertTrue(saved_path.name.endswith(".jpg"))

    def test_severity_analyzer_reports_expected_levels(self):
        self.assertEqual(SeverityAnalyzer.compute(0.2, 2, 1), "LOW")
        self.assertEqual(SeverityAnalyzer.compute(0.5, 2, 1), "MEDIUM")
        self.assertEqual(SeverityAnalyzer.compute(0.8, 3, 2), "HIGH")


if __name__ == "__main__":
    unittest.main()
