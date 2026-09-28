"""
Tests for NexGuard Event Logger Module
"""

import unittest
import json
import tempfile
from pathlib import Path
from src.event_logger import DetectionEventLogger


class TestEventLogger(unittest.TestCase):

    def test_log_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "events.jsonl"
            logger = DetectionEventLogger(str(log_file))

            detections = [
                {"class_name": "car", "confidence": 0.9},
                {"class_name": "car", "confidence": 0.8},
                {"class_name": "person", "confidence": 0.95}
            ]

            logger.log_detection(frame_id=1, detections=detections, fps=30.0)

            self.assertTrue(log_file.exists())
            lines = log_file.read_text().strip().split("\n")
            self.assertEqual(len(lines), 1)

            data = json.loads(lines[0])
            self.assertEqual(data["frame_id"], 1)
            self.assertEqual(data["total_objects"], 3)
            self.assertEqual(data["counts"]["car"], 2)
            self.assertEqual(data["counts"]["person"], 1)


if __name__ == "__main__":
    unittest.main()
