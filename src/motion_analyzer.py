"""
NexGuard Motion Analysis Engine
Analyzes track displacement, velocity, direction changes, sudden deceleration,
motion reversal, and stationary transitions over temporal window buffers.
"""

from typing import Dict, List, Tuple, Any, Optional
import math
from utils.geometry import (
    compute_euclidean_distance,
    compute_vector_magnitude,
    compute_angle_change
)
from src.tracker import TrackedObject


class TrackMotionMetrics:
    """Calculated motion metrics for a single tracked object across temporal history."""

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.displacement: float = 0.0
        self.current_speed: float = 0.0
        self.previous_speed: float = 0.0
        self.speed_drop_ratio: float = 0.0
        self.direction_change_deg: float = 0.0
        self.is_sudden_deceleration: bool = False
        self.is_abrupt_turn: bool = False
        self.is_stationary: bool = False
        self.is_post_collision_stopped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "displacement": round(self.displacement, 2),
            "current_speed": round(self.current_speed, 2),
            "speed_drop_ratio": round(self.speed_drop_ratio, 2),
            "direction_change_deg": round(self.direction_change_deg, 1),
            "is_sudden_deceleration": self.is_sudden_deceleration,
            "is_abrupt_turn": self.is_abrupt_turn,
            "is_stationary": self.is_stationary,
            "is_post_collision_stopped": self.is_post_collision_stopped
        }


class MotionAnalyzer:
    """Analyzes object tracking history for anomalous motion patterns."""

    def __init__(
        self,
        speed_drop_threshold: float = 0.50,
        direction_change_threshold: float = 45.0
    ):
        self.speed_drop_threshold = speed_drop_threshold
        self.direction_change_threshold = direction_change_threshold

    def analyze_track(self, track: TrackedObject) -> TrackMotionMetrics:
        """Computes motion metrics for a single track based on its temporal history."""
        metrics = TrackMotionMetrics(track.track_id)
        history = track.center_history

        if len(history) < 2:
            metrics.current_speed = track.speed
            metrics.is_stationary = (track.speed < 0.5)
            return metrics

        # Total displacement across history window
        metrics.displacement = compute_euclidean_distance(history[0], history[-1])
        metrics.current_speed = track.speed

        # Calculate previous velocity segment if available
        if len(history) >= 4:
            p0, p1 = history[-4], history[-2]
            prev_vx = p1[0] - p0[0]
            prev_vy = p1[1] - p0[1]
            metrics.previous_speed = compute_vector_magnitude((prev_vx, prev_vy))

            curr_vx, curr_vy = track.velocity
            metrics.direction_change_deg = compute_angle_change((prev_vx, prev_vy), (curr_vx, curr_vy))

            # Speed reduction ratio
            if metrics.previous_speed > 2.0:
                metrics.speed_drop_ratio = (metrics.previous_speed - metrics.current_speed) / metrics.previous_speed
                if metrics.speed_drop_ratio >= self.speed_drop_threshold:
                    metrics.is_sudden_deceleration = True

            if metrics.direction_change_deg >= self.direction_change_threshold and metrics.previous_speed > 2.0:
                metrics.is_abrupt_turn = True

        # Stationary transition check
        metrics.is_stationary = (metrics.current_speed < 0.5)
        if metrics.is_stationary and len(history) >= 5 and metrics.previous_speed > 3.0:
            metrics.is_post_collision_stopped = True

        return metrics

    def analyze_all_tracks(self, tracks: List[TrackedObject]) -> Dict[int, TrackMotionMetrics]:
        """Analyzes all active tracks in frame."""
        return {track.track_id: self.analyze_track(track) for track in tracks}
