"""
Tests for NexGuard Utilities
"""

import unittest
import tempfile
from pathlib import Path
from src.utils import detect_device, get_device_info, validate_file_path, ensure_dir, generate_snapshot_filename


class TestUtils(unittest.TestCase):

    def test_detect_device(self):
        device = detect_device()
        self.assertIn(device, ["cpu", "cuda"])

    def test_get_device_info(self):
        dev_type, dev_name = get_device_info()
        self.assertIn(dev_type, ["CPU", "CUDA"])
        self.assertIsInstance(dev_name, str)

    def test_validate_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.assertFalse(validate_file_path(str(tmp_path / "nonexistent.mp4")))

            test_file = tmp_path / "sample.mp4"
            test_file.write_text("dummy video content")

            self.assertTrue(validate_file_path(str(test_file)))
            self.assertTrue(validate_file_path(str(test_file), allowed_extensions=(".mp4", ".avi")))
            self.assertFalse(validate_file_path(str(test_file), allowed_extensions=(".jpg", ".png")))

    def test_ensure_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "test_dir" / "sub_dir"
            created = ensure_dir(str(target_dir))
            self.assertTrue(created.exists())
            self.assertTrue(created.is_dir())

    def test_generate_snapshot_filename(self):
        filename = generate_snapshot_filename("test")
        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".jpg"))


if __name__ == "__main__":
    unittest.main()
