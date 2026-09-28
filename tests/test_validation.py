"""
NexGuard System & Asset Validation Tests
Validates synthetic asset creation, video stream initialization, and detector metadata.
"""

import unittest
import tempfile
from pathlib import Path

from src.video import VideoFileInput, ImageFileInput
from src.detector import NexGuardDetector
from assets.sample.generate_samples import create_sample_image, create_sample_video


class TestSystemValidation(unittest.TestCase):

    def test_sample_asset_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_p = Path(tmpdir) / "test_img.jpg"
            vid_p = Path(tmpdir) / "test_vid.mp4"

            res_img = create_sample_image(img_p)
            res_vid = create_sample_video(vid_p, num_frames=5)

            self.assertTrue(res_img.exists())
            self.assertTrue(res_vid.exists())

            # Test ImageFileInput with generated image
            img_src = ImageFileInput(str(res_img))
            self.assertTrue(img_src.is_opened)
            img_src.release()

            # Test VideoFileInput with generated video
            vid_src = VideoFileInput(str(res_vid))
            self.assertTrue(vid_src.is_opened)
            self.assertEqual(vid_src.total_frames, 5)
            vid_src.release()

    def test_detector_metadata_validation(self):
        detector = NexGuardDetector(model_path="yolov8n.pt", confidence_threshold=0.35, device="cpu")
        meta = detector.get_metadata()
        self.assertEqual(meta["confidence_threshold"], 0.35)
        self.assertEqual(meta["device"], "cpu")
        self.assertIn("class_count", meta)


if __name__ == "__main__":
    unittest.main()
