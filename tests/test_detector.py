"""
Tests for NexGuard YOLO Detector Engine
"""

import unittest
import numpy as np
from src.detector import NexGuardDetector


class TestDetector(unittest.TestCase):

    def test_detector_initialization(self):
        detector = NexGuardDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.3,
            iou_threshold=0.5,
            device="cpu"
        )
        metadata = detector.get_metadata()
        self.assertEqual(metadata["model_path"], "yolov8n.pt")
        self.assertEqual(metadata["confidence_threshold"], 0.3)
        self.assertEqual(metadata["device"], "cpu")
        self.assertIn("class_count", metadata)

    def test_predict_invalid_input(self):
        detector = NexGuardDetector(model_path="dummy.pt", device="cpu")
        # None input
        self.assertEqual(detector.predict(None), [])
        # Non-array input
        self.assertEqual(detector.predict("invalid_image_path"), [])

    def test_predict_empty_frame(self):
        detector = NexGuardDetector(model_path="dummy.pt", device="cpu")
        # Black frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.predict(frame)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
