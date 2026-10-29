"""
NexGuard — General-Purpose Helpers
====================================
Reusable utility functions used across the pipeline.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def bbox_iou(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Intersection over Union for two (x1, y1, x2, y2) boxes."""
    try:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
        union = a1 + a2 - inter
        val = inter / union if union > 1e-6 else 0.0
        return val if math.isfinite(val) else 0.0
    except Exception:
        return 0.0


def bbox_centroid_dist(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Euclidean distance between the centres of two bounding boxes."""
    try:
        cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
        cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
        d = math.hypot(cx2 - cx1, cy2 - cy1)
        return d if math.isfinite(d) else 9999.0
    except Exception:
        return 9999.0


def bbox_diagonal(b: Tuple[float, float, float, float]) -> float:
    """Diagonal length of a bounding box."""
    return math.hypot(b[2] - b[0], b[3] - b[1])


def heading_diff(h1: float, h2: float) -> float:
    """Smallest angular difference between two headings (result 0–180°)."""
    if not (math.isfinite(h1) and math.isfinite(h2)):
        return 0.0
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS estimator."""

    def __init__(self, window: int = 30) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """Record a new frame timestamp."""
        now = time.monotonic()
        self._timestamps.append(now)
        cutoff = now - 2.0  # keep only last 2 seconds
        self._timestamps = [t for t in self._timestamps if t >= cutoff]
        # Also enforce window cap
        if len(self._timestamps) > self._window:
            self._timestamps = self._timestamps[-self._window:]

    @property
    def fps(self) -> float:
        """Current FPS estimate."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def seconds_to_timestamp(seconds: float) -> str:
    """Converts a float seconds value to HH:MM:SS.ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def now_iso() -> str:
    """Current local time as ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))
