"""
NexGuard — Video Input Source Abstraction
==========================================
Provides a unified interface for all video input types:
  - Webcam (integer device index)
  - Local video file (.mp4, .avi, .mkv, etc.)
  - RTSP / IP camera stream (rtsp://)
  - HTTP stream (http://)

Usage:
    source = VideoSource("0")          # webcam
    source = VideoSource("video.mp4")  # file
    source = VideoSource("rtsp://...")  # IP camera

    with source:
        while source.is_open():
            ok, frame = source.read()
            if not ok:
                break
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from nexguard.utils.logging import get_logger

log = get_logger("nexguard.input")


class VideoSource:
    """
    Unified video input abstraction supporting webcam, file, RTSP, and HTTP sources.

    Attributes:
        source_str: The original source string as passed by the user.
        source_type: One of 'webcam', 'file', 'rtsp', 'http'.
        cap: Underlying cv2.VideoCapture instance.
        frame_width: Frame width in pixels (set after opening).
        frame_height: Frame height in pixels (set after opening).
        fps: Source frames-per-second (set after opening).
        total_frames: Total frame count (valid for file sources only).
    """

    def __init__(self, source: str, reconnect_attempts: int = 3) -> None:
        """
        Args:
            source:             Video source string. Numeric strings (e.g. '0') are
                                treated as webcam device indices.
            reconnect_attempts: Number of reconnection retries for stream sources.
        """
        self.source_str = str(source)
        self.reconnect_attempts = reconnect_attempts
        self.cap: Optional[cv2.VideoCapture] = None

        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = 30.0
        self.total_frames: int = 0

        self.source_type = self._detect_type()

    def _detect_type(self) -> str:
        s = self.source_str.lower()
        if self.source_str.isdigit():
            return "webcam"
        if s.startswith("rtsp://"):
            return "rtsp"
        if s.startswith("http://") or s.startswith("https://"):
            return "http"
        return "file"

    def open(self) -> None:
        """Opens the video source. Raises RuntimeError if unable to open."""
        if self.source_type == "webcam":
            index = int(self.source_str)
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open webcam at device index {index}. "
                    "Check that a camera is connected and accessible."
                )
        elif self.source_type == "file":
            path = Path(self.source_str)
            if not path.exists():
                raise FileNotFoundError(
                    f"Video file not found: {self.source_str}. "
                    "Check the path and try again."
                )
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot open video file: {self.source_str}. "
                    "The file may be corrupt or in an unsupported format."
                )
        elif self.source_type in ("rtsp", "http"):
            self.cap = cv2.VideoCapture(self.source_str)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Cannot connect to stream: {self.source_str}. "
                    "Verify the URL, network connectivity, and camera credentials."
                )
        else:
            raise ValueError(f"Unrecognized source type for: {self.source_str}")

        # Capture metadata
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if raw_fps and raw_fps > 0 else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log.info(
            f"Opened {self.source_type} source: {self.source_str} "
            f"[{self.frame_width}x{self.frame_height} @ {self.fps:.1f} FPS]"
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next frame.

        Returns:
            (True, frame) on success; (False, None) on failure or end of stream.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ok, frame = self.cap.read()

        if not ok and self.source_type in ("rtsp", "http"):
            log.warning(f"Stream read failed — attempting reconnect ({self.source_str})")
            for attempt in range(1, self.reconnect_attempts + 1):
                time.sleep(1.0)
                self.cap.open(self.source_str)
                if self.cap.isOpened():
                    ok, frame = self.cap.read()
                    if ok:
                        log.info(f"Stream reconnected on attempt {attempt}")
                        break
                log.warning(f"Reconnect attempt {attempt} failed")

        return ok, frame if ok else None

    def is_open(self) -> bool:
        """Returns True if the video source is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Releases the video capture resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        log.info(f"Released source: {self.source_str}")

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoSource(type={self.source_type!r}, source={self.source_str!r}, "
            f"resolution={self.frame_width}x{self.frame_height}, fps={self.fps:.1f})"
        )
