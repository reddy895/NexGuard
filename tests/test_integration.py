"""
End-to-end integration test using real sample video and real YOLOv8n weights.
Tests the complete capture -> inference -> tracking -> event evaluation pipeline.
"""

from pathlib import Path
import pytest

from config import cfg, MAX_FPS
from nexguard.accident.accident_detector import AccidentDetector
from nexguard.detection.detector import ObjectDetector
from nexguard.input.source import VideoSource
from nexguard.tracking.tracker import ObjectTracker
from nexguard.utils.rate_limiter import MonotonicRateLimiter


def test_real_video_and_model_integration():
    sample_clip = Path("test_clips/traffic.mp4")
    model_path = Path("yolov8n.pt")

    if not sample_clip.exists() or not model_path.exists():
        pytest.skip("Integration test requires test_clips/traffic.mp4 and yolov8n.pt")

    source = VideoSource(str(sample_clip))
    source.open()
    assert source.is_open(), "Failed to open traffic.mp4 sample clip"

    detector = ObjectDetector(model_path=str(model_path), conf=0.25)
    assert detector.model_ready is True

    tracker = ObjectTracker()
    acc_detector = AccidentDetector(threshold=0.60, confirm_frames=4)
    rate_limiter = MonotonicRateLimiter(max_fps=MAX_FPS)

    frames_processed = 0
    inferences_run = 0
    max_frames_to_test = 20

    events_detected = []

    while source.is_open() and frames_processed < max_frames_to_test:
        ok, frame = source.read()
        if not ok or frame is None:
            break
        frames_processed += 1

        # Check rate limiter
        if rate_limiter.should_process_frame():
            rate_limiter.record_inference_start()
            inferences_run += 1

            # Run real YOLO inference
            detections = detector.detect(frame)
            assert isinstance(detections, list)

            # Run tracking
            tracks = tracker.update(detections)
            assert isinstance(tracks, list)

            # Run accident event analysis
            event = acc_detector.update(tracks)
            if event:
                events_detected.append(event)

            rate_limiter.record_inference_complete()

    source.release()

    assert frames_processed > 0
    assert inferences_run > 0
    # Strict maximum FPS enforcement check:
    assert rate_limiter.current_fps <= MAX_FPS + 0.1

    # In traffic.mp4, traffic is normal flowing traffic.
    # A generic COCO model with heuristic kinematics should NOT falsely flag a major crash in normal flow.
    assert len(events_detected) == 0, "Normal traffic should not trigger false positive accident alerts"
