"""
Tests for NexGuard Video Input Source Abstraction
"""

import unittest
import tempfile
import cv2
import numpy as np
from pathlib import Path
from src.video import VideoFileInput, ImageFileInput, WebcamInput


class TestVideoInput(unittest.TestCase):

    def test_image_file_input_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_image.jpg"
            # Create synthetic test image
            dummy_img = np.full((240, 320, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(img_path), dummy_img)

            inp = ImageFileInput(str(img_path))
            self.assertTrue(inp.is_opened)
            info = inp.get_info()
            self.assertEqual(info["width"], 320)
            self.assertEqual(info["height"], 240)
            self.assertEqual(info["source_type"], "image")

            ret, frame = inp.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (240, 320, 3))

            inp.release()
            self.assertFalse(inp.is_opened)

    def test_image_file_input_invalid(self):
        inp = ImageFileInput("nonexistent_path.png")
        self.assertFalse(inp.is_opened)
        ret, frame = inp.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)

    def test_video_file_input_invalid(self):
        inp = VideoFileInput("nonexistent_video.mp4")
        self.assertFalse(inp.is_opened)
        ret, frame = inp.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)


if __name__ == "__main__":
    unittest.main()
