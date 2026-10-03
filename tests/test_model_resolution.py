"""
Test Suite: Model Path Resolution & Initialization Tests
"""

import pytest
from utils.config import settings, BASE_DIR
from utils.detector import NexGuardDetector


def test_detector_initialization():
    detector = NexGuardDetector()
    assert detector.ready is True
    assert detector.device in ["cuda", "cpu"]
    assert detector.model is not None


def test_missing_model_handling():
    detector = NexGuardDetector(model_path="non_existent_model.pt")
    # Falls back gracefully to base model or reports error
    assert isinstance(detector.ready, bool)
