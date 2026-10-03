"""
NexGuard Performance Optimization & Real FPS Monitoring Module
Provides frame skipping logic, mode presets, and empirical FPS computation.
"""

import time
from typing import Tuple, Dict, Any
from config import config


class PerformanceMonitor:
    """Tracks real processed FPS and manages frame skip intervals."""

    def __init__(self, mode: str = "balanced"):
        self.mode = mode
        self.frame_counter = 0
        self.processed_counter = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.current_fps = 0.0

    def set_mode(self, mode: str):
        """Updates performance mode (accuracy, balanced, performance)."""
        config.apply_performance_mode(mode)
        self.mode = mode

    def should_process_frame(self) -> bool:
        """Determines if current frame should be sent for YOLO inference based on frame skip."""
        self.frame_counter += 1
        skip = config.process_every_n_frames
        if skip <= 1:
            return True
        return (self.frame_counter % skip) == 0

    def tick_processed(self):
        """Increments processed frames counter and updates empirical FPS."""
        self.processed_counter += 1
        now = time.time()
        elapsed = now - self.start_time
        if elapsed > 0.5:
            self.current_fps = self.processed_counter / elapsed

    def get_fps(self) -> float:
        """Returns empirical processed FPS."""
        return round(self.current_fps, 1)

    def reset(self):
        """Resets timer and counters."""
        self.frame_counter = 0
        self.processed_counter = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.current_fps = 0.0
