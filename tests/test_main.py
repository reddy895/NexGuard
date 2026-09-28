"""
Tests for NexGuard Main Application Functions
"""

import unittest
from io import StringIO
import sys
from config import NexGuardConfig
from main import print_banner, show_initialization_checks


class TestMain(unittest.TestCase):

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
