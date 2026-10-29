"""
Tests — System Robustness, Pipeline Errors, and Edge Cases
===========================================================
Validates:
  - Missing model weights handling
  - Invalid video paths handling
  - Camera initialization failures
  - Returning to terminal menu after closing video / pressing Q
  - Handling unknown or uncertain events
  - Zero detections propagation through tracking and event logic
  - Config validation and device resolution
  - Geometric utility calculations (IoU, distance, centroid)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from config import NexGuardConfig, cfg
from nexguard.accident.accident_detector import AccidentDetector
from nexguard.accident.event import IncidentManager
from nexguard.accident.severity import SeverityClassifier
from nexguard.detection.detector import ObjectDetector
from nexguard.models.schemas import Detection, DetectionMode, EventType, ModelType, Severity
from nexguard.tracking.tracker import ObjectTracker
from nexguard.utils.utils import bbox_centroid_dist, bbox_diagonal, bbox_iou, heading_diff
from overlay import Overlay
from video_processor import VideoProcessor


# ===========================================================================
# 1. Missing Model Weights Handling
# ===========================================================================

class TestMissingModelWeights:
    def test_missing_weights_fails_safely_without_crash(self):
        detector = ObjectDetector(model_path="nonexistent_path_to_model_12345.pt")
        assert detector.model_ready is False
        diag = detector.get_diagnostic_info()
        assert diag.ready is False
        assert diag.model_type == ModelType.UNKNOWN
        assert diag.class_count == 0
        assert "Failed to load YOLO model" in diag.error_message or "not found" in diag.error_message.lower()

    def test_detect_on_unready_model_returns_empty_list(self):
        detector = ObjectDetector(model_path="nonexistent_weights.pt")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame)
        assert dets == []
        assert isinstance(dets, list)


# ===========================================================================
# 2. Invalid Video Paths & Camera Initialization Failures
# ===========================================================================

class TestInputSourceFailures:
    @pytest.fixture
    def processor(self):
        detector = MagicMock(spec=ObjectDetector)
        tracker = MagicMock(spec=ObjectTracker)
        acc_detector = MagicMock(spec=AccidentDetector)
        severity = MagicMock(spec=SeverityClassifier)
        incidents = MagicMock(spec=IncidentManager)
        overlay = MagicMock(spec=Overlay)
        return VideoProcessor(detector, tracker, acc_detector, severity, incidents, overlay)

    def test_invalid_video_path_fails_gracefully(self, processor, capsys):
        processor.run_file("nonexistent_video_file_98765.mp4")
        captured = capsys.readouterr()
        assert "[ERROR] File not found" in captured.out

    def test_corrupt_video_file_fails_gracefully(self, processor, capsys, tmp_path):
        fake_video = tmp_path / "corrupt.mp4"
        fake_video.write_bytes(b"not a valid video stream")
        processor.run_file(str(fake_video))
        captured = capsys.readouterr()
        assert "[ERROR] Cannot open video file" in captured.out or "Playback finished" in captured.out

    @patch("cv2.VideoCapture")
    def test_camera_init_failure_reports_clear_error(self, mock_cap_cls, processor, capsys):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        processor.run_webcam(camera_index=99)
        captured = capsys.readouterr()
        assert "[ERROR] Cannot open webcam at index 99" in captured.out


# ===========================================================================
# 3. Returning to Terminal Menu After Video / Stream Closes
# ===========================================================================

class TestCleanReturnToMenu:
    @patch("cv2.destroyAllWindows")
    @patch("cv2.waitKey", return_value=ord("q"))
    @patch("cv2.VideoCapture")
    def test_user_quits_with_q_releases_resources_and_returns(
        self, mock_cap_cls, mock_wait_key, mock_destroy, capsys
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 640
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap_cls.return_value = mock_cap

        detector = MagicMock(spec=ObjectDetector)
        detector.detect.return_value = []
        tracker = MagicMock(spec=ObjectTracker)
        tracker.update.return_value = []
        acc_detector = MagicMock(spec=AccidentDetector)
        acc_detector.update.return_value = None
        severity = MagicMock(spec=SeverityClassifier)
        incidents = MagicMock(spec=IncidentManager)
        incidents.active_incidents = []
        overlay = MagicMock(spec=Overlay)
        overlay.draw.return_value = dummy_frame
        overlay.show.return_value = ord("q")

        proc = VideoProcessor(detector, tracker, acc_detector, severity, incidents, overlay)
        with patch.object(Path, "exists", return_value=True):
            proc.run_file("test.mp4")

        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()
        captured = capsys.readouterr()
        assert "Returning to menu" in captured.out

    @patch("cv2.destroyAllWindows")
    @patch("cv2.VideoCapture")
    def test_video_end_of_file_returns_cleanly(
        self, mock_cap_cls, mock_destroy, capsys
    ):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10.0
        # End of stream immediately
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        detector = MagicMock(spec=ObjectDetector)
        tracker = MagicMock(spec=ObjectTracker)
        acc_detector = MagicMock(spec=AccidentDetector)
        severity = MagicMock(spec=SeverityClassifier)
        incidents = MagicMock(spec=IncidentManager)
        incidents.active_incidents = []
        overlay = MagicMock(spec=Overlay)

        proc = VideoProcessor(detector, tracker, acc_detector, severity, incidents, overlay)
        with patch.object(Path, "exists", return_value=True):
            proc.run_file("test.mp4")

        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()
        captured = capsys.readouterr()
        assert "Playback finished. Returning to menu." in captured.out


# ===========================================================================
# 4. Handling Unknown or Uncertain Events
# ===========================================================================

class TestUncertainAndUnknownEvents:
    def test_borderline_confidence_returns_uncertain(self):
        classifier = SeverityClassifier()
        # Mock event with borderline confidence (0.46) on trained model
        mock_event = MagicMock()
        mock_event.is_model_confirmed = True
        mock_event.detection_mode = DetectionMode.TRAINED_MODEL
        mock_event.confidence = 0.46
        mock_event.event_type = EventType.CONFIRMED_MODEL_EVENT
        mock_event.track_ids = [1, 2]

        result = classifier.classify(
            event=mock_event,
            involved_tracks=[],
            all_tracks=[],
            person_involvements=[],
            persistence_frames=1,
        )
        assert result.severity == Severity.UNCERTAIN
        assert result.severity.value == "UNCERTAIN"
        assert any("insufficient" in r.lower() or "supporting" in r.lower() for r in result.reasons)

    def test_unknown_event_type_handled_safely(self):
        classifier = SeverityClassifier()
        mock_event = MagicMock()
        mock_event.is_model_confirmed = False
        mock_event.detection_mode = DetectionMode.HEURISTIC
        mock_event.confidence = 0.30
        mock_event.event_type = EventType.UNKNOWN
        mock_event.track_ids = [1]

        result = classifier.classify(
            event=mock_event,
            involved_tracks=[],
            all_tracks=[],
            person_involvements=[],
            persistence_frames=1,
        )
        assert result.severity in (Severity.LOW, Severity.UNCERTAIN)
        assert isinstance(result.score, float)


# ===========================================================================
# 5. Zero Detections Propagation
# ===========================================================================

class TestZeroDetectionsPropagation:
    def test_empty_detections_preserve_tracker_and_accident_state(self):
        tracker = ObjectTracker()
        acc_detector = AccidentDetector()

        for _ in range(5):
            tracks = tracker.update([])
            assert tracks == []
            event = acc_detector.update(tracks)
            assert event is None


# ===========================================================================
# 6. Configuration & Utilities Validation
# ===========================================================================

class TestConfigAndUtils:
    def test_config_validation_raises_on_invalid_thresholds(self):
        cfg_bad = NexGuardConfig(confidence_threshold=1.5)
        with pytest.raises(ValueError, match="NEXGUARD_CONFIDENCE"):
            cfg_bad.validate()

        cfg_bad_iou = NexGuardConfig(iou_threshold=-0.1)
        with pytest.raises(ValueError, match="NEXGUARD_IOU_THRESHOLD"):
            cfg_bad_iou.validate()

    def test_resolve_device_cpu_fallback(self):
        config = NexGuardConfig(device="cpu")
        assert config.resolve_device() == "cpu"

    def test_bounding_box_and_distance_math(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (0.0, 0.0, 10.0, 10.0)
        assert bbox_iou(box1, box2) == 1.0

        box3 = (20.0, 20.0, 30.0, 30.0)
        assert bbox_iou(box1, box3) == 0.0
        assert bbox_centroid_dist(box1, box2) == 0.0
        assert bbox_diagonal(box1) == pytest.approx(14.142, rel=1e-3)
        assert heading_diff(0.0, 90.0) == 90.0
