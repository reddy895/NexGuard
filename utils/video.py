"""
NexGuard Video Stream & Frame Processing Utilities
"""

import time
from typing import Tuple, Optional
import cv2
import numpy as np


class VideoStreamHandler:
    """Helper wrapper around OpenCV VideoCapture for robust frame reading and FPS tracking."""

    def __init__(self, source: str | int):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.start_time = time.time()
        self.frame_count = 0
        self.last_frame_time = time.time()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened():
            return False, None
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.frame_count += 1
            self.last_frame_time = time.time()
        return ret, frame

    def calculate_actual_fps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed <= 0 or self.frame_count == 0:
            return 0.0
        return self.frame_count / elapsed

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def resize_frame(frame: np.ndarray, width: int = 1280, height: int = 720) -> np.ndarray:
    """Resizes frame to target display dimensions maintaining clean aspect ratio."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if w == width and h == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
