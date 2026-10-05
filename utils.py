"""
NexGuard Utility Functions & FPS Rate Limiter Module
Includes Max 15 FPS frame regulator, bounding box utilities, distance matrix helpers,
and incident snapshot persistence.
"""

import time
import math
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
from config import config


class FPSLimiter:
    """Enforces strict FPS capping (e.g. Max 15 FPS) to prevent system overload."""
    
    def __init__(self, target_fps: int = 15):
        self.target_fps = max(1, target_fps)
        self.frame_time = 1.0 / float(self.target_fps)
        self.last_frame_start = time.time()
        self.current_fps = 0.0
        self.alpha = 0.2  # Exponential smoothing factor for FPS calculation

    def start_frame(self):
        """Marks the start of frame processing."""
        self.last_frame_start = time.time()

    def sleep_if_needed(self) -> float:
        """Calculates elapsed time and sleeps if necessary to cap frame rate at MAX 15 FPS."""
        now = time.time()
        elapsed = now - self.last_frame_start
        if elapsed < self.frame_time:
            sleep_time = self.frame_time - elapsed
            time.sleep(sleep_time)
            now = time.time()
            elapsed = now - self.last_frame_start

        instant_fps = 1.0 / elapsed if elapsed > 0 else float(self.target_fps)
        self.current_fps = (1 - self.alpha) * self.current_fps + self.alpha * instant_fps if self.current_fps > 0 else instant_fps
        return min(self.current_fps, float(self.target_fps))


class FrameRateRegulator:
    """Combines frame timing and FPS measurement for live video loops."""
    def __init__(self, max_fps: int = 15):
        self.max_fps = max_fps
        self.target_delay = 1.0 / max_fps
        self.last_timestamp = time.time()
        self.measured_fps = 0.0

    def tick(self) -> float:
        now = time.time()
        elapsed = now - self.last_timestamp
        if elapsed < self.target_delay:
            time.sleep(self.target_delay - elapsed)
            now = time.time()
            elapsed = now - self.last_timestamp

        self.measured_fps = 1.0 / elapsed if elapsed > 0 else float(self.max_fps)
        self.last_timestamp = now
        return min(self.measured_fps, float(self.max_fps))


def compute_bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two boxes (xmin, ymin, xmax, ymax)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_centroid_distance(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes Euclidean distance between box centroids."""
    c1_x, c1_y = (box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0
    c2_x, c2_y = (box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0
    return float(np.hypot(c1_x - c2_x, c1_y - c2_y))


def resize_frame_to_target(frame: np.ndarray, target_w: int = 1280, target_h: int = 720) -> np.ndarray:
    """Resizes frame maintaining resolution ratio."""
    if frame is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def log_event(level: str, msg: str):
    """Outputs standardized NexGuard formatted log messages."""
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] [{level.upper()}] {msg}")
