"""
Tests for NexGuard Main Application Functions
"""

import unittest
from io import StringIO
import sys
import subprocess
from unittest.mock import patch
from config import NexGuardConfig
from main import choose_media_file, print_banner, show_initialization_checks


class TestMain(unittest.TestCase):

    def test_choose_media_file_uses_zenity(self):
        result = subprocess.CompletedProcess(
            args=["zenity"], returncode=0, stdout="/tmp/traffic.mp4\n", stderr=""
        )
        with patch("main.shutil.which", return_value="/usr/bin/zenity"), patch(
            "main.subprocess.run", return_value=result
        ) as run_picker:
            selected = choose_media_file("video")

        self.assertEqual(selected, "/tmp/traffic.mp4")
        command = run_picker.call_args.args[0]
        self.assertIn("--file-selection", command)
        self.assertTrue(any("*.mp4" in argument for argument in command))

    def test_print_banner(self):
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            print_banner()
            output = out.getvalue()
            self.assertIn("NEXGUARD", output)
            self.assertIn("EDGE AI SURVEILLANCE SYSTEM", output)
        finally:
            sys.stdout = saved_stdout

    def test_show_initialization_checks(self):
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            cfg = NexGuardConfig()
            res = show_initialization_checks(cfg)
            self.assertTrue(res)
            output = out.getvalue()
            self.assertIn("Python environment", output)
            self.assertIn("YOLO engine", output)
            self.assertIn("Configuration", output)
        finally:
            sys.stdout = saved_stdout


if __name__ == "__main__":
    unittest.main()
