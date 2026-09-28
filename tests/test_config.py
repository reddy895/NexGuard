"""
Tests for NexGuard Configuration System
"""

import unittest
from config import NexGuardConfig, DEFAULT_CONFIG


class TestConfig(unittest.TestCase):

    def test_default_config_initialization(self):
        cfg = NexGuardConfig()
        self.assertEqual(cfg.model_path, "yolov8n.pt")
        self.assertEqual(cfg.confidence_threshold, 0.25)
        self.assertEqual(cfg.camera_index, 0)
        self.assertIn(cfg.device, ["cpu", "cuda"])

    def test_threshold_clamping(self):
        cfg_high = NexGuardConfig(confidence_threshold=1.5, iou_threshold=2.0)
        self.assertEqual(cfg_high.confidence_threshold, 1.0)
        self.assertEqual(cfg_high.iou_threshold, 1.0)

        cfg_low = NexGuardConfig(confidence_threshold=-0.5, iou_threshold=-1.0)
        self.assertEqual(cfg_low.confidence_threshold, 0.0)
        self.assertEqual(cfg_low.iou_threshold, 0.0)

    def test_to_dict_and_from_dict(self):
        original = NexGuardConfig(model_path="models/yolov8s.pt", confidence_threshold=0.5, camera_index=1)
        d = original.to_dict()
        self.assertEqual(d["model_path"], "models/yolov8s.pt")
        self.assertEqual(d["confidence_threshold"], 0.5)
        self.assertEqual(d["camera_index"], 1)

        restored = NexGuardConfig.from_dict(d)
        self.assertEqual(restored.model_path, "models/yolov8s.pt")
        self.assertEqual(restored.confidence_threshold, 0.5)
        self.assertEqual(restored.camera_index, 1)


if __name__ == "__main__":
    unittest.main()
