"""
Test Suite 2: Temporal Accident Detection Logic & Multi-Frame Confirmation
"""

import pytest
from utils.detector import DetectionResult
from utils.tracker import NexGuardTracker
from utils.accident_analyzer import AccidentDetector


def test_single_frame_does_not_declare_accident():
    detector = AccidentDetector()
    tracker = NexGuardTracker()

    # Frame 1: High proximity cars
    d1 = DetectionResult(class_id=2, class_name="car", confidence=0.90, bbox=[100, 100, 200, 200])
    d2 = DetectionResult(class_id=2, class_name="car", confidence=0.90, bbox=[110, 110, 210, 210])

    tracks = tracker.update([d1, d2])
    res = detector.analyze_frame(tracks)

    # Must require temporal evidence across multiple frames
    assert res.is_accident is False
    assert detector.temporal_evidence_counter == 1


def test_multi_frame_temporal_confirmation():
    detector = AccidentDetector()
    tracker = NexGuardTracker()

    # Simulate 3 consecutive overlapping frames
    for _ in range(3):
        d1 = DetectionResult(class_id=2, class_name="car", confidence=0.90, bbox=[100, 100, 200, 200])
        d2 = DetectionResult(class_id=2, class_name="car", confidence=0.90, bbox=[105, 105, 205, 205])
        tracks = tracker.update([d1, d2])
        res = detector.analyze_frame(tracks)

    assert res.is_accident is True
    assert res.severity.level in ["MEDIUM", "HIGH", "CRITICAL"]
