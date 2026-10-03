"""
Unit Tests for NexGuard Collision Analyzer with Deterministic Synthetic Track Scenarios
"""

import pytest
from src.detector import DetectionObject
from src.tracker import TrackedObject
from src.motion_analyzer import MotionAnalyzer
from src.collision_analyzer import CollisionAnalyzer


def test_normal_passing_vehicles_no_collision():
    analyzer = CollisionAnalyzer(proximity_threshold=120.0, iou_threshold=0.15, candidate_score_threshold=0.35)
    motion_analyzer = MotionAnalyzer()

    # Car 1 in Lane 1 moving right
    det1 = DetectionObject(bbox=[100, 100, 150, 150], confidence=0.9, class_id=2, class_name="car")
    track1 = TrackedObject(track_id=1, detection=det1)
    
    # Car 2 in Lane 2 moving right (parallel passing)
    det2 = DetectionObject(bbox=[100, 250, 150, 300], confidence=0.9, class_id=2, class_name="car")
    track2 = TrackedObject(track_id=2, detection=det2)

    # Move parallel for several frames
    for step in range(5):
        det1_step = DetectionObject(bbox=[100 + step*15, 100, 150 + step*15, 150], confidence=0.9, class_id=2, class_name="car")
        det2_step = DetectionObject(bbox=[100 + step*15, 250, 150 + step*15, 300], confidence=0.9, class_id=2, class_name="car")
        track1.update(det1_step)
        track2.update(det2_step)

    tracks = [track1, track2]
    motion_metrics = motion_analyzer.analyze_all_tracks(tracks)
    candidates, max_score, involved = analyzer.analyze_collisions(tracks, motion_metrics)

    assert len(candidates) == 0, "Normal passing vehicles should NOT generate collision candidates."
    assert max_score < 0.35


def test_synthetic_headon_collision_scenario():
    analyzer = CollisionAnalyzer(proximity_threshold=120.0, iou_threshold=0.15, candidate_score_threshold=0.35)
    motion_analyzer = MotionAnalyzer()

    # Car 1 moving right
    det1 = DetectionObject(bbox=[100, 100, 160, 160], confidence=0.9, class_id=2, class_name="car")
    track1 = TrackedObject(track_id=1, detection=det1)

    # Car 2 moving left directly towards Car 1
    det2 = DetectionObject(bbox=[250, 100, 310, 160], confidence=0.9, class_id=2, class_name="car")
    track2 = TrackedObject(track_id=2, detection=det2)

    # Simulate approach and overlap collision
    positions1 = [(120, 100), (140, 100), (160, 100), (175, 100)]
    positions2 = [(230, 100), (210, 100), (185, 100), (175, 100)]

    for p1, p2 in zip(positions1, positions2):
        d1 = DetectionObject(bbox=[p1[0], p1[1], p1[0]+60, p1[1]+60], confidence=0.9, class_id=2, class_name="car")
        d2 = DetectionObject(bbox=[p2[0], p2[1], p2[0]+60, p2[1]+60], confidence=0.9, class_id=2, class_name="car")
        track1.update(d1)
        track2.update(d2)

    tracks = [track1, track2]
    motion_metrics = motion_analyzer.analyze_all_tracks(tracks)
    candidates, max_score, involved = analyzer.analyze_collisions(tracks, motion_metrics)

    assert len(candidates) >= 1
    assert set(involved) == {1, 2}
    assert max_score >= 0.35
