"""
Test Suite 3: File Input & Path Validation Tests
"""

import pytest
from pathlib import Path
from utils.config import BASE_DIR, UPLOADS_DIR, INCIDENTS_DIR


def test_directory_existence():
    assert UPLOADS_DIR.exists()
    assert INCIDENTS_DIR.exists()


def test_nonexistent_file_handling():
    fake_path = BASE_DIR / "non_existent_file.jpg"
    assert not fake_path.exists()
