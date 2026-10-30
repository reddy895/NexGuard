"""
NexGuard — False-Positive Suppression Engine
============================================
Suppresses common traffic false alarms using physical and behavioral rules:
1. Parallel Lane Driving: Vehicles travelling side-by-side at matched speeds.
2. Normal Traffic Braking: Gradual stop at signals or stoplights without overlap.
3. Parked Vehicles: Stationary cars that never had prior movement.
4. Pedestrian Proximity: Pedestrians walking alongside normal traffic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
from nexguard.models.schemas import Track
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.suppression")

@dataclass
class SuppressionVerdict:
    should_suppress: bool
    reason: str = ""

class FalsePositiveFilter:
    """
    Evaluates kinematic contexts to filter out non-accident traffic occurrences.
    """

    @staticmethod
    def is_parallel_lane_driving(t1: Track, t2: Track, iou: float) -> SuppressionVerdict:
        """
        Suppresses false alarms when vehicles drive parallel in adjacent lanes.
        """
        if iou >= 0.08:
            return SuppressionVerdict(should_suppress=False)

        # Both moving at substantial speed
        if t1.speed > 2.0 and t2.speed > 2.0:
            speed_ratio = min(t1.speed, t2.speed) / max(t1.speed, t2.speed)
            heading_diff = abs((t1.heading - t2.heading + 180) % 360 - 180)
            # Parallel heading (< 20 deg) and similar speed (> 70%)
            if heading_diff < 20.0 and speed_ratio > 0.70:
                return SuppressionVerdict(
                    should_suppress=True,
                    reason=f"Parallel lane flow: heading_diff={heading_diff:.1f}deg speed_ratio={speed_ratio:.0%}",
                )

        return SuppressionVerdict(should_suppress=False)

    @staticmethod
    def is_normal_signal_stop(track: Track) -> SuppressionVerdict:
        """
        Suppresses single vehicle stop if deceleration was gradual over time.
        """
        if len(track.speed_history) < 8:
            return SuppressionVerdict(should_suppress=False)

        # Gradual deceleration without abrupt drop
        drops = [
            (track.speed_history[i] - track.speed_history[i + 1])
            for i in range(len(track.speed_history) - 1)
        ]
        max_step_drop = max(drops, default=0.0)

        # If smooth stop (no step drop > 3.0 px/frame)
        if max_step_drop < 2.5 and track.speed < 0.5:
            return SuppressionVerdict(
                should_suppress=True,
                reason="Smooth controlled signal stop",
            )

        return SuppressionVerdict(should_suppress=False)

    @staticmethod
    def is_parked_vehicle(track: Track) -> SuppressionVerdict:
        """
        Suppresses tracks that have never moved since spawn.
        """
        if track.age > 10 and max(track.speed_history) < 0.5:
            return SuppressionVerdict(
                should_suppress=True,
                reason="Stationary parked vehicle with zero velocity history",
            )
        return SuppressionVerdict(should_suppress=False)
