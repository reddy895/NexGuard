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
