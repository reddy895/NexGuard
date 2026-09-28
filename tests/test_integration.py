"""
NexGuard End-to-End Pipeline Integration Test
Validates integration across VideoInput, Detector, Display, and EventLogger modules.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.video import ImageFileInput, VideoFileInput
from src.detector import NexGuardDetector
from src.display import draw_detections, draw_overlay_stats
from src.event_logger import DetectionEventLogger
from assets.sample.generate_samples import create_sample_image, create_sample_video


class TestIntegrationPipeline(unittest.TestCase):

    def test_e2e_image_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "sample.jpg"
            create_sample_image(img_path)

            input_src = ImageFileInput(str(img_path))
            self.assertTrue(input_src.is_opened)

            detector = NexGuardDetector(model_path="dummy.pt", device="cpu")
            event_logger = DetectionEventLogger(str(Path(tmpdir) / "events.jsonl"))

            ret, frame = input_src.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)

            detections = detector.predict(frame)
            self.assertIsInstance(detections, list)

            annotated = draw_detections(frame, detections)
            display_frame = draw_overlay_stats(
                annotated,
                fps=25.0,
                frame_count=1,
                object_count=len(detections),
                device="CPU",
                model_name="yolov8n.pt"
            )
            self.assertEqual(display_frame.shape, frame.shape)

            event_logger.log_detection(frame_id=1, detections=detections, fps=25.0)
            input_src.release()

    def test_e2e_video_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / "sample.mp4"
            create_sample_video(vid_path, num_frames=10)

            input_src = VideoFileInput(str(vid_path))
            self.assertTrue(input_src.is_opened)

            detector = NexGuardDetector(model_path="dummy.pt", device="cpu")
            frame_cnt = 0

            while input_src.is_opened:
                ret, frame = input_src.read_frame()
                if not ret:
                    break
                frame_cnt += 1
                detections = detector.predict(frame)
                annotated = draw_detections(frame, detections)
                self.assertEqual(annotated.shape, frame.shape)

            self.assertEqual(frame_cnt, 10)
            input_src.release()


if __name__ == "__main__":
    unittest.main()
