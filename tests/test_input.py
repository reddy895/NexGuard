"""
Unit Tests for NexGuard Input Path and Source Validation
"""

import pytest
from pathlib import Path
from utils.validation import (
    validate_image_path,
    validate_video_path,
    validate_webcam_index,
    validate_rtsp_url
)


def test_validate_image_path_missing(tmp_path):
    missing_file = str(tmp_path / "non_existent.jpg")
    valid, msg, path_obj = validate_image_path(missing_file)
    assert not valid
    assert "not found" in msg.lower()
    assert path_obj is None


def test_validate_image_path_invalid_extension(tmp_path):
    invalid_file = tmp_path / "document.pdf"
    invalid_file.write_text("dummy")
    valid, msg, path_obj = validate_image_path(str(invalid_file))
    assert not valid
    assert "unsupported image extension" in msg.lower()


def test_validate_video_path_missing(tmp_path):
    missing_vid = str(tmp_path / "non_existent.mp4")
    valid, msg, path_obj = validate_video_path(missing_vid)
    assert not valid
    assert "not found" in msg.lower()
    assert path_obj is None


def test_validate_webcam_index_invalid():
    valid, msg, idx = validate_webcam_index("invalid_index")
    assert not valid
    assert "invalid webcam index" in msg.lower()


def test_validate_rtsp_url_invalid_format():
    valid, msg, url = validate_rtsp_url("ftp://invalid_stream")
    assert not valid
    assert "must start with" in msg.lower()
