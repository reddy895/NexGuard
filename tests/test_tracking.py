"""
Unit Tests for NexGuard Object Tracker
"""

import pytest
from src.detector import DetectionObject
from src.tracker import ObjectTracker, TrackedObject


def test_tracker_creation_and_update():
    tracker = ObjectTracker()

    det1 = DetectionObject(bbox=[100, 100, 200, 200], confidence=0.85, class_id=2, class_name="car")
    tracks = tracker.update([det1])

    assert len(tracks) == 1
    t1 = tracks[0]
    assert t1.track_id == 1
    assert t1.class_name == "car"
    assert t1.center == (150.0, 150.0)
    assert t1.frames_tracked == 1

    # Update with second frame position
    det2 = DetectionObject(bbox=[110, 100, 210, 200], confidence=0.88, class_id=2, class_name="car")
    tracks_f2 = tracker.update([det2])

    assert len(tracks_f2) == 1
    t1_updated = tracks_f2[0]
    assert t1_updated.track_id == 1
    assert t1_updated.frames_tracked == 2
    assert t1_updated.velocity == (10.0, 0.0)
    assert t1_updated.speed == 10.0
