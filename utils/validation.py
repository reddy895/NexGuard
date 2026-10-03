"""
NexGuard Input & Dataset Validation Utilities
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Any
import cv2

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}


def validate_image_path(path_str: str) -> Tuple[bool, str, Optional[Path]]:
    """
    Validates image file path existence, extension, and decodability.
    Returns (is_valid, error_message, Path_object).
    """
    if not path_str or not path_str.strip():
        return False, "ERROR: Image file path cannot be empty.", None

    clean_path = path_str.strip().strip('"').strip("'")
    path = Path(clean_path).expanduser().resolve()

    if not path.exists():
        return False, f"ERROR: Image file not found: {path}", None

    if not path.is_file():
        return False, f"ERROR: Path is not a file: {path}", None

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return False, f"ERROR: Unsupported image extension '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}", None

    # Verify OpenCV can decode the image
    img = cv2.imread(str(path))
    if img is None:
        return False, f"ERROR: Unable to decode image file: {path}", None

    return True, "", path


def validate_video_path(path_str: str) -> Tuple[bool, str, Optional[Path]]:
    """
    Validates video file path existence, extension, and OpenCV opener.
    Returns (is_valid, error_message, Path_object).
    """
    if not path_str or not path_str.strip():
        return False, "ERROR: Video file path cannot be empty.", None

    clean_path = path_str.strip().strip('"').strip("'")
    path = Path(clean_path).expanduser().resolve()

    if not path.exists():
        return False, f"ERROR: Video file not found: {path}", None

    if not path.is_file():
        return False, f"ERROR: Path is not a file: {path}", None

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return False, f"ERROR: Unsupported video extension '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}", None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False, f"ERROR: Unable to open video file: {path}", None

    cap.release()
    return True, "", path


def validate_webcam_index(idx_input: Any) -> Tuple[bool, str, int]:
    """
    Validates webcam device index.
    Returns (is_valid, error_message, index_int).
    """
    try:
        idx = int(idx_input)
        if idx < 0:
            return False, "ERROR: Webcam index must be a non-negative integer.", 0
    except (ValueError, TypeError):
        return False, f"ERROR: Invalid webcam index '{idx_input}'. Must be an integer.", 0

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return False, f"ERROR: Unable to access webcam at index {idx}.", idx

    cap.release()
    return True, "", idx


def validate_rtsp_url(rtsp_str: str) -> Tuple[bool, str, str]:
    """
    Validates RTSP stream URL format and connectivity.
    Returns (is_valid, error_message, clean_url).
    """
    if not rtsp_str or not rtsp_str.strip():
        return False, "ERROR: RTSP URL cannot be empty.", ""

    url = rtsp_str.strip()
    if not (url.lower().startswith("rtsp://") or url.lower().startswith("http://") or url.lower().startswith("https://")):
        return False, "ERROR: URL must start with rtsp://, http://, or https://", ""

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return False, f"ERROR: Unable to connect to stream URL: {url}", url

    cap.release()
    return True, "", url


def validate_accident_dataset(dataset_dir: Path) -> Tuple[bool, str]:
    """
    Validates YOLO dataset structure:
    training/dataset/
        ├── images/ (train/, val/)
        ├── labels/ (train/, val/)
        └── data.yaml
    """
    if not dataset_dir.exists():
        return False, f"Dataset directory missing: {dataset_dir}"

    train_imgs_dir = dataset_dir / "images" / "train"
    val_imgs_dir = dataset_dir / "images" / "val"
    train_lbls_dir = dataset_dir / "labels" / "train"
    val_lbls_dir = dataset_dir / "labels" / "val"
    data_yaml = dataset_dir / "data.yaml"

    if not train_imgs_dir.exists() or not list(train_imgs_dir.glob("*.*")):
        return False, f"No training images found in: {train_imgs_dir}"

    if not val_imgs_dir.exists() or not list(val_imgs_dir.glob("*.*")):
        return False, f"No validation images found in: {val_imgs_dir}"

    if not train_lbls_dir.exists() or not list(train_lbls_dir.glob("*.*")):
        return False, f"No training labels found in: {train_lbls_dir}"

    if not val_lbls_dir.exists() or not list(val_lbls_dir.glob("*.*")):
        return False, f"No validation labels found in: {val_lbls_dir}"

    if not data_yaml.exists():
        return False, f"Dataset descriptor missing: {data_yaml}"

    return True, "Dataset structure is valid."
