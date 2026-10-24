"""
NexGuard — Monotonic Rate Limiter Unit Tests
=============================================
Tests deterministic enforcement of the strict 15 FPS ceiling using a mocked clock.
Validates:
1. Never exceeding 15 inferences in any rolling 1.0-second interval.
2. Inter-frame spacing and burst prevention.
3. Live stream non-blocking frame drops.
4. Slow inference naturally decreasing throughput.
"""

import sys
import unittest
from typing import List

sys.path.insert(0, ".")

from nexguard.utils.rate_limiter import MonotonicRateLimiter, StreamInferenceLock


class MockClock:
    """Mocked monotonic clock and sleep function for deterministic rate limiter tests."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time
        self.sleep_calls: List[float] = []

    def now(self) -> float:
        return self.current_time

    def sleep(self, duration: float) -> None:
        if duration > 0:
            self.sleep_calls.append(duration)
            self.current_time += duration

    def advance(self, delta: float) -> None:
        self.current_time += delta


class TestRateLimiter(unittest.TestCase):
    """MonotonicRateLimiter unit tests."""

    def test_never_exceeds_15_inferences_in_rolling_one_second(self):
        """Strict mathematical guarantee: count(inferences in any [t - 1.0s, t]) <= 15."""
        clock = MockClock()
        limiter = MonotonicRateLimiter(
            max_fps=15,
            window_seconds=1.0,
            clock_fn=clock.now,
            sleep_fn=clock.sleep,
        )

        inferences: List[float] = []
        # Simulate attempting 45 rapid inference requests
        for _ in range(45):
            allowed = limiter.acquire(blocking=True)
            self.assertTrue(allowed)
            inferences.append(clock.now())
            # Simulate a 10ms inference execution
            clock.advance(0.010)

        self.assertEqual(len(inferences), 45)

        # Audit every single rolling 1-second interval
        for t in inferences:
            window_inferences = [ts for ts in inferences if (t - 1.0) < ts <= t]
            self.assertLessEqual(
                len(window_inferences),
                15,
                f"Rolling window ending at {t:.3f} had {len(window_inferences)} inferences (limit is 15)",
            )

    def test_inter_frame_spacing_prevents_bursting(self):
        """15 FPS requires ~66.7ms minimum interval to prevent millisecond bursts."""
        clock = MockClock()
        limiter = MonotonicRateLimiter(
            max_fps=15,
            window_seconds=1.0,
            clock_fn=clock.now,
            sleep_fn=clock.sleep,
        )

        t1 = clock.now()
        self.assertTrue(limiter.acquire(blocking=True))
        # Instant inference (0ms)
        self.assertTrue(limiter.acquire(blocking=True))
        t2 = clock.now()

        # The limiter must pace frames by sleeping at least min_interval (~66.7ms)
        self.assertGreaterEqual(t2 - t1, 0.065)

    def test_live_source_drops_excess_frames(self):
        """Live video sources must drop frames when saturated to prevent latency buildup."""
        clock = MockClock()
        limiter = MonotonicRateLimiter(
            max_fps=15,
            window_seconds=1.0,
            clock_fn=clock.now,
            sleep_fn=clock.sleep,
        )

        allowed_count = 0
        skipped_count = 0

        # Feed 30 incoming camera frames in 0.2 seconds (rapid capture)
        for _ in range(30):
            allowed, reason = limiter.should_process_frame(is_live=True, drop_on_saturation=True)
            if allowed:
                allowed_count += 1
                clock.advance(0.010)
            else:
                skipped_count += 1
                clock.advance(0.005)

        self.assertGreater(skipped_count, 0)
        self.assertLessEqual(allowed_count, 15)
        self.assertEqual(limiter.total_frames_seen, 30)
        self.assertEqual(limiter.total_skipped, skipped_count)

    def test_stream_inference_lock(self):
        """Only one inference worker may run per stream simultaneously."""
        stream_id = "camera_0"
        lock1 = StreamInferenceLock.get_lock(stream_id)
        lock2 = StreamInferenceLock.get_lock(stream_id)
        self.assertIs(lock1, lock2)

        acquired = lock1.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            # Second attempt without release must fail
            acquired_second = lock2.acquire(blocking=False)
            self.assertFalse(acquired_second)
        finally:
            lock1.release()

        StreamInferenceLock.release_stream(stream_id)


if __name__ == "__main__":
    unittest.main()
