"""
Tests for ObjectDetector, model validation, and class mapping.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexguard.detection.detector import ObjectDetector
from nexguard.detection.validator import inspect_model_weights
from nexguard.models.schemas import ModelType


def test_missing_model_weights_reports_unready():
    non_existent = "models/non_existent_weights_12345.pt"
    detector = ObjectDetector(model_path=non_existent)
    assert detector.model_ready is False
    assert detector.diagnostic_info.ready is False
    assert "No such file" in detector.diagnostic_info.error_message or "not found" in detector.diagnostic_info.error_message.lower()


def test_validator_detects_generic_coco_model():
    with patch("pathlib.Path.exists", return_value=True), patch("ultralytics.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_instance.names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}
        mock_yolo.return_value = mock_instance

        diag = inspect_model_weights("yolov8n.pt")
        assert diag.model_type == ModelType.GENERIC_DETECTOR
        assert len(diag.accident_classes_found) == 0
        assert diag.class_count == 6
        assert diag.classes[2] == "car"


def test_validator_detects_accident_trained_model():
    with patch("pathlib.Path.exists", return_value=True), patch("ultralytics.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_instance.names = {0: "severe_accident", 1: "moderate_accident", 2: "car", 3: "truck"}
        mock_yolo.return_value = mock_instance

        diag = inspect_model_weights("accident_detector.pt")
        assert diag.model_type in (ModelType.ACCIDENT_DETECTOR, ModelType.HYBRID_DETECTOR)
        assert len(diag.accident_classes_found) >= 2
        assert "severe_accident" in diag.accident_classes_found


def test_detector_class_names_mapping():
    with patch("pathlib.Path.exists", return_value=True), patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {0: "person", 1: "car", 2: "bus", 3: "accident"}
        mock_instance.return_value = [MagicMock()]
        mock_yolo_cls.return_value = mock_instance

        detector = ObjectDetector(model_path="dummy_path.pt")
        assert detector.model_ready is True
        assert detector.class_names == {0: "person", 1: "car", 2: "bus", 3: "accident"}
        diag = detector.get_diagnostic_info()
        assert diag.class_count == 4
        assert "accident" in diag.accident_classes_found


def test_zero_detections_in_frame():
    import numpy as np
    from nexguard.tracking.tracker import ObjectTracker

    with patch("pathlib.Path.exists", return_value=True), patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.names = {0: "person", 1: "car"}
        # Mock empty results with 0 boxes
        mock_box = MagicMock()
        mock_box.__len__.return_value = 0
        mock_box.xyxy = []
        mock_box.conf = []
        mock_box.cls = []
        mock_res = MagicMock()
        mock_res.boxes = mock_box
        mock_instance.return_value = [mock_res]
        mock_yolo_cls.return_value = mock_instance

        detector = ObjectDetector(model_path="dummy.pt")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(blank_frame)
        assert isinstance(dets, list)
        assert len(dets) == 0

        # Propagate empty detections to tracker
        tracker = ObjectTracker()
        active_tracks = tracker.update(dets)
        assert len(active_tracks) == 0


def test_empty_tensor_does_not_crash_pipeline():
    import numpy as np
    from nexguard.accident.accident_detector import AccidentDetector
    from nexguard.tracking.tracker import ObjectTracker

    tracker = ObjectTracker()
    acc_detector = AccidentDetector()

    # Pass 5 consecutive frames with zero detections
    for _ in range(5):
        tracks = tracker.update([])
        assert tracks == []
        event = acc_detector.update(tracks)
        assert event is None


def test_validate_accident_dataset_missing():
    from nexguard.detection.validator import validate_accident_dataset

    valid, msg, meta = validate_accident_dataset("nonexistent_datasets/data.yaml")
    assert valid is False
    assert "NEXGUARD ERROR: ACCIDENT DATASET REQUIRED" in msg
    assert meta["status"] == "MISSING"


def test_validate_accident_dataset_valid(tmp_path):
    import yaml
    from nexguard.detection.validator import validate_accident_dataset

    yaml_file = tmp_path / "data.yaml"
    data = {
        "train": str(tmp_path / "train"),
        "val": str(tmp_path / "val"),
        "nc": 2,
        "names": {0: "moderate_accident", 1: "severe_accident"}
    }
    with open(yaml_file, "w") as f:
        yaml.dump(data, f)

    valid, msg, meta = validate_accident_dataset(str(yaml_file))
    assert valid is True
    assert meta["status"] == "VALID"
    assert meta["num_classes"] == 2
    assert "moderate_accident" in meta["classes"].values()


def test_strict_accident_model_flags_generic_detector():
    from config import cfg

    with patch("pathlib.Path.exists", return_value=True), patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        # Generic COCO detector without accident classes
        mock_instance.names = {0: "person", 1: "car", 2: "truck"}
        mock_instance.return_value = [MagicMock()]
        mock_yolo_cls.return_value = mock_instance

        # Temporarily enable strict accident model check
        original_strict = cfg.strict_accident_model
        try:
            cfg.strict_accident_model = True
            detector = ObjectDetector(model_path="yolov8n.pt")
            assert detector.model_ready is False
            assert "NEXGUARD ERROR: ACCIDENT DATASET REQUIRED" in detector.diagnostic_info.error_message
        finally:
            cfg.strict_accident_model = original_strict


def test_pipeline_propagation_kinematic_collision():
    from nexguard.accident.accident_detector import AccidentDetector
    from nexguard.models.schemas import Detection, EventType
    from nexguard.tracking.tracker import ObjectTracker

    tracker = ObjectTracker(iou_threshold=0.15)
    acc_detector = AccidentDetector(confirm_frames=3, threshold=0.50)

    # Frame 1: Cars far apart, moving toward each other
    dets_f1 = [
        Detection(bbox=(100, 100, 150, 150), class_name="car", confidence=0.85),
        Detection(bbox=(300, 100, 350, 150), class_name="car", confidence=0.88),
    ]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 2
    assert acc_detector.update(tracks_f1) is None

    # Frame 2: Moving closer fast (closing speed)
    dets_f2 = [
        Detection(bbox=(160, 100, 210, 150), class_name="car", confidence=0.85),
        Detection(bbox=(240, 100, 290, 150), class_name="car", confidence=0.88),
    ]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 2
    assert acc_detector.update(tracks_f2) is None

    # Frame 3: Near contact, high approach speed
    dets_f3 = [
        Detection(bbox=(190, 100, 240, 150), class_name="car", confidence=0.85),
        Detection(bbox=(210, 100, 260, 150), class_name="car", confidence=0.88),
    ]
    tracks_f3 = tracker.update(dets_f3)
    acc_detector.update(tracks_f3)

    # Frames 4-7: Collision overlap & sudden stop (motion halted)
    confirmed_event = None
    for f in range(4, 9):
        dets_overlap = [
            Detection(bbox=(195, 100, 245, 150), class_name="car", confidence=0.85),
            Detection(bbox=(205, 100, 255, 150), class_name="car", confidence=0.88),
        ]
        tracks = tracker.update(dets_overlap)
        event = acc_detector.update(tracks)
        if event:
            confirmed_event = event

    assert confirmed_event is not None
    assert confirmed_event.accident_detected is True
    assert confirmed_event.confidence >= 0.50
    assert confirmed_event.event_type in (EventType.VEHICLE_COLLISION, EventType.POSSIBLE_ACCIDENT)
    assert len(confirmed_event.track_ids) >= 1


def test_pipeline_propagation_direct_model_event():
    from nexguard.accident.accident_detector import AccidentDetector
    from nexguard.models.schemas import Detection, DetectionMode, EventType
    from nexguard.tracking.tracker import ObjectTracker

    tracker = ObjectTracker()
    acc_detector = AccidentDetector(confirm_frames=2, threshold=0.50)

    confirmed_event = None
    for _ in range(4):
        dets = [
            Detection(
                bbox=(200, 200, 350, 350),
                class_name="severe_accident",
                confidence=0.89,
                is_accident=True,
            )
        ]
        tracks = tracker.update(dets)
        event = acc_detector.update(tracks)
        if event:
            confirmed_event = event

    assert confirmed_event is not None
    assert confirmed_event.is_model_confirmed is True
    assert confirmed_event.event_type == EventType.CONFIRMED_MODEL_EVENT
    assert confirmed_event.detection_mode == DetectionMode.TRAINED_MODEL
