"""
NexGuard — Monotonic Inference Rate Limiter
============================================
Enforces a strict maximum of 15 processed inference operations per rolling
one-second window using a monotonic clock.

Design Principles:
- Hard rate limit: Never begin more than MAX_FPS (15) inferences in any rolling 1.0s interval.
- Monotonic clock: Immune to system clock adjustments or daylight savings shifts.
- Pacing without bursting: Enforces minimum inter-frame spacing (1.0 / MAX_FPS) to prevent
  bursting 15 frames at millisecond intervals and freezing for the remainder of the second.
- Graceful degradation: When inference is slow (> 1/15s), the rate naturally decreases
  below 15 FPS without queue buildup.
- Clock injection: Accepts custom clock and sleep callables for deterministic unit testing.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Callable, Deque, Optional, Tuple

from config import MAX_FPS
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.rate_limiter")


class MonotonicRateLimiter:
    """
    Precision rate limiter enforcing a strict maximum of MAX_FPS inference calls
    in any rolling one-second interval.
    """

    def __init__(
        self,
        max_fps: int = MAX_FPS,
        window_seconds: float = 1.0,
        clock_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_fps: int = min(max_fps, MAX_FPS)
        self.window_seconds: float = float(window_seconds)
        self.min_interval: float = 1.0 / float(self.max_fps) if self.max_fps > 0 else 0.0

        self._clock: Callable[[], float] = clock_fn
        self._sleep: Callable[[float], None] = sleep_fn

        self._inference_timestamps: Deque[float] = collections.deque()
        self._last_inference_start: float = -1.0
        self._lock: threading.Lock = threading.Lock()

        # Telemetry
        self._total_frames_seen: int = 0
        self._total_inferences: int = 0
        self._total_skipped: int = 0
        self._slow_inference_count: int = 0
        self._total_inference_time_s: float = 0.0

    def purge_old_timestamps(self, now: float) -> None:
        """Removes timestamps outside the rolling window."""
        cutoff = now - self.window_seconds
        while self._inference_timestamps and self._inference_timestamps[0] <= cutoff:
            self._inference_timestamps.popleft()

    def can_process_now(self) -> bool:
        """
        Returns True if an inference operation can start immediately without violating
        the rolling 1-second limit or bursting constraints.
        """
        with self._lock:
            now = self._clock()
            self.purge_old_timestamps(now)
            if len(self._inference_timestamps) >= self.max_fps:
                return False
            if self._last_inference_start > 0 and (now - self._last_inference_start) < (self.min_interval * 0.95):
                return False
            return True

    def acquire(self, blocking: bool = True) -> bool:
        """
        Requests permission to begin an inference operation.

        If blocking=True:
            Sleeps the required duration to respect both the inter-frame pacing
            and rolling 1-second limit, ensuring no more than max_fps inferences
            occur in any rolling one-second interval.

        If blocking=False:
            Returns True immediately if allowed, False if capacity is exceeded.

        Returns:
            True if inference may proceed; False otherwise.
        """
        with self._lock:
            while True:
                now = self._clock()
                self.purge_old_timestamps(now)

                # Check rolling 1-second capacity
                if len(self._inference_timestamps) >= self.max_fps:
                    if not blocking:
                        self._total_skipped += 1
                        return False
                    # Must wait until oldest timestamp in window expires
                    earliest = self._inference_timestamps[0]
                    sleep_time = (earliest + self.window_seconds) - now
                    if sleep_time > 0:
                        self._sleep(sleep_time)
                    continue

                # Check inter-frame pacing (prevent bursting)
                if self._last_inference_start > 0:
                    elapsed_since_last = now - self._last_inference_start
                    if elapsed_since_last < self.min_interval:
                        needed = self.min_interval - elapsed_since_last
                        if not blocking:
                            self._total_skipped += 1
                            return False
                        self._sleep(needed)
                        continue

                # Record successful acquisition
                now_acquired = self._clock()
                self.purge_old_timestamps(now_acquired)
                self._inference_timestamps.append(now_acquired)
                self._last_inference_start = now_acquired
                self._total_inferences += 1
                return True

    def record_frame_seen(self) -> None:
        """Increments total frames received from the video source."""
        with self._lock:
            self._total_frames_seen += 1

    def record_frame_skipped(self, reason: str = "rate_limit") -> None:
        """Records a skipped frame."""
        with self._lock:
            self._total_skipped += 1

    def should_process_frame(
        self,
        is_live: bool = False,
        drop_on_saturation: bool = True,
    ) -> Tuple[bool, str]:
        """
        Determines whether a frame should undergo model inference.

        Policy:
        - Live sources (webcam, RTSP, HTTP): Non-blocking check. If capacity
          is saturated or inter-frame pacing is violated, skips the frame
          immediately to prevent queue buildup and latency accumulation.
        - Prerecorded files: Uses pacing to process at up to MAX_FPS, preserving
          frame progression and timestamps.

        Returns:
            (allow_inference: bool, status_reason: str)
        """
        self.record_frame_seen()

        if is_live and drop_on_saturation:
            allowed = self.acquire(blocking=False)
            if allowed:
                return True, "PROCESSED_LIVE"
            return False, "SKIPPED_LIVE_RATE_LIMIT"

        # Prerecorded file or blocking mode: pace smoothly up to max_fps
        allowed = self.acquire(blocking=True)
        if allowed:
            return True, "PROCESSED_FILE"
        return False, "SKIPPED_CAPACITY"

    @property
    def current_fps(self) -> float:
        """
        Calculates actual processed inference FPS over the rolling window.
        """
        with self._lock:
            now = self._clock()
            self.purge_old_timestamps(now)
            count = len(self._inference_timestamps)
            if count == 0:
                return 0.0
            return float(count) / self.window_seconds

    @property
    def total_frames_seen(self) -> int:
        return self._total_frames_seen

    def record_inference_completion(self, duration_s: float) -> None:
        """
        Records the duration of an inference operation.
        If duration_s exceeds the minimum frame interval (1/15s), logs and records
        that hardware compute is naturally throttling FPS below MAX_FPS.
        """
        with self._lock:
            self._total_inference_time_s += duration_s
            if duration_s > self.min_interval:
                self._slow_inference_count += 1
                log.debug(
                    f"Inference duration ({duration_s*1000:.1f}ms) exceeded pacing interval "
                    f"({self.min_interval*1000:.1f}ms). Effective FPS naturally reduces."
                )

    @property
    def total_inferences(self) -> int:
        return self._total_inferences

    @property
    def total_skipped(self) -> int:
        return self._total_skipped

    @property
    def slow_inference_count(self) -> int:
        return self._slow_inference_count

    @property
    def mean_inference_time_ms(self) -> float:
        if self._total_inferences == 0:
            return 0.0
        return (self._total_inference_time_s / float(self._total_inferences)) * 1000.0

    @property
    def skip_ratio(self) -> float:
        if self._total_frames_seen == 0:
            return 0.0
        return float(self._total_skipped) / float(self._total_frames_seen)

    def metrics_summary(self) -> dict:
        """Returns telemetry snapshot of the rate limiter."""
        return {
            "current_inference_fps": round(self.current_fps, 1),
            "max_fps_limit": self.max_fps,
            "total_frames_seen": self._total_frames_seen,
            "total_inferences": self._total_inferences,
            "total_skipped": self._total_skipped,
            "slow_inferences": self._slow_inference_count,
            "mean_inference_time_ms": round(self.mean_inference_time_ms, 1),
            "skip_ratio": round(self.skip_ratio, 3),
        }

    def reset(self) -> None:
        """Resets all rate limiter counters and history."""
        with self._lock:
            self._inference_timestamps.clear()
            self._last_inference_start = -1.0
            self._total_frames_seen = 0
            self._total_inferences = 0
            self._total_skipped = 0
            self._slow_inference_count = 0
            self._total_inference_time_s = 0.0
