"""
Tests for NexGuard Display Module
"""

import unittest
import numpy as np
from src.display import get_class_color, draw_detections, draw_grid, draw_overlay_stats


class TestDisplay(unittest.TestCase):

    def test_get_class_color(self):
        color1 = get_class_color(0)
        color2 = get_class_color(1)
        self.assertEqual(len(color1), 3)
        self.assertIsInstance(color1[0], int)
        # Verify color consistency for same ID
        self.assertEqual(get_class_color(0), color1)
        # Verify different colors for different IDs
        self.assertNotEqual(color1, color2)

    def test_draw_detections(self):
        # Create dummy black frame 480x640x3
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"box": [50, 50, 200, 200], "confidence": 0.88, "class_id": 0, "class_name": "person"},
            {"box": [250, 100, 400, 300], "confidence": 0.95, "class_id": 2, "class_name": "car"},
        ]
        annotated = draw_detections(frame, detections)
        self.assertEqual(annotated.shape, frame.shape)
        # Ensure image frame was modified (pixels non-zero)
        self.assertTrue(np.any(annotated > 0))

    def test_draw_grid(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_grid(frame)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.any(annotated[:, 213, :] > 0))
        self.assertTrue(np.any(annotated[160, :, :] > 0))

    def test_draw_overlay_stats(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_overlay_stats(
            frame,
            fps=28.5,
            frame_count=150,
            object_count=3,
            device="CUDA",
            model_name="yolov8n.pt",
            paused=False
        )
        self.assertEqual(annotated.shape, frame.shape)
        self.assertTrue(np.any(annotated > 0))


if __name__ == "__main__":
    unittest.main()
