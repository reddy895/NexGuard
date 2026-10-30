"""
NexGuard — Trajectory and Kinematic Analysis Engine
===================================================
Analyzes multi-frame motion trajectories for collision dynamics:
- Time-to-Collision (TTC) and Distance of Closest Approach (DCA)
- Acceleration / Deceleration profiles and jerk detection
- Angular heading divergence and post-impact trajectory deflection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from nexguard.models.schemas import Track

@dataclass
class TrajectoryMetrics:
    speed: float
    acceleration: float
    heading_deg: float
    curvature: float
    is_sudden_stop: bool
    is_deflected: bool

class TrajectoryAnalyzer:
    """
    Computes trajectory kinematics and collision risks between moving tracks.
    """

    @staticmethod
    def compute_acceleration(speed_history: List[float], dt: float = 1.0) -> float:
        if len(speed_history) < 2:
            return 0.0
        return (speed_history[-1] - speed_history[-2]) / max(dt, 1e-6)

    @staticmethod
    def compute_time_to_collision(t1: Track, t2: Track) -> Optional[float]:
        """
        Estimates Time-to-Collision (TTC) in frames assuming linear motion.
        Returns None if objects are not closing or moving in parallel/diverging.
        """
        dx = t2.cx - t1.cx
        dy = t2.cy - t1.cy
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return 0.0

        v1x, v1y = t1.velocity if t1.velocity else (0.0, 0.0)
        v2x, v2y = t2.velocity if t2.velocity else (0.0, 0.0)
        rel_vx = v1x - v2x
        rel_vy = v1y - v2y

        # Project relative velocity along line of separation
        closing_speed = (rel_vx * dx + rel_vy * dy) / dist
        if closing_speed <= 0.5:  # Diverging or stationary
            return None

        ttc = dist / closing_speed
        return float(ttc)

    @staticmethod
    def analyze_trajectory(track: Track) -> TrajectoryMetrics:
        hist = track.history
        speed_hist = track.speed_history

        accel = TrajectoryAnalyzer.compute_acceleration(speed_hist)

        # Heading variation
        heading = track.heading
        curvature = 0.0
        if len(hist) >= 3:
            p1, p2, p3 = hist[-3], hist[-2], hist[-1]
            a1 = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            a2 = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]))
            curvature = abs((a2 - a1 + 180) % 360 - 180)

        sudden_stop = False
        if len(speed_hist) >= 4:
            recent_max = max(speed_hist[:-1])
            if recent_max > 4.0 and track.speed < 1.0:
                sudden_stop = True

        deflected = curvature > 40.0 and track.speed > 2.0

        return TrajectoryMetrics(
            speed=track.speed,
            acceleration=accel,
            heading_deg=heading,
            curvature=curvature,
            is_sudden_stop=sudden_stop,
            is_deflected=deflected,
        )
