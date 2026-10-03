"""
Unit Tests for NexGuard Motion Analyzer
"""

import pytest
from src.detector import DetectionObject
from src.tracker import TrackedObject
from src.motion_analyzer import MotionAnalyzer, TrackMotionMetrics


def test_motion_analyzer_speed_drop_and_deceleration():
    analyzer = MotionAnalyzer(speed_drop_threshold=0.50, direction_change_threshold=45.0)

    det = DetectionObject(bbox=[100, 100, 200, 200], confidence=0.9, class_id=2, class_name="car")
    track = TrackedObject(track_id=1, detection=det)

    # Simulate trajectory moving fast then suddenly stopping
    positions = [
        (100.0, 100.0),
        (120.0, 100.0),
        (140.0, 100.0),
        (160.0, 100.0),
        (160.0, 100.0)  # Complete stop
    ]

    for pos in positions[1:]:
        det_next = DetectionObject(
            bbox=[pos[0] - 50, pos[1] - 50, pos[0] + 50, pos[1] + 50],
            confidence=0.9,
            class_id=2,
            class_name="car"
        )
        track.update(det_next)

    metrics = analyzer.analyze_track(track)
    assert metrics.track_id == 1
    assert metrics.is_sudden_deceleration is True
    assert metrics.is_stationary is True
