"""
NexGuard Collision Candidate Analyzer Module
Evaluates spatial proximity, bounding box overlap (IoU), relative approach velocity,
and motion anomalies between pairs of tracked vehicles.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from config import config
from utils.geometry import compute_euclidean_distance, compute_iou, compute_vector_magnitude
from src.tracker import TrackedObject
from src.motion_analyzer import TrackMotionMetrics, MotionAnalyzer


class CollisionCandidate:
    """Represents a potential collision event between two tracked vehicles."""

    def __init__(
        self,
        v1: TrackedObject,
        v2: TrackedObject,
        score: float,
        iou: float,
        distance: float,
        reasons: List[str]
    ):
        self.track_id_1: int = v1.track_id
        self.track_id_2: int = v2.track_id
        self.class_1: str = v1.class_name
        self.class_2: str = v2.class_name
        self.score: float = round(score, 4)
        self.iou: float = round(iou, 4)
        self.distance: float = round(distance, 2)
        self.reasons: List[str] = reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicles": [self.track_id_1, self.track_id_2],
            "classes": [self.class_1, self.class_2],
            "score": self.score,
            "iou": self.iou,
            "distance": self.distance,
            "reasons": self.reasons
        }


class CollisionAnalyzer:
    """Evaluates tracked vehicle interactions to identify collision candidates."""

    def __init__(
        self,
        proximity_threshold: float = None,
        iou_threshold: float = None,
        candidate_score_threshold: float = 0.35
    ):
        self.proximity_threshold = proximity_threshold or config.proximity_threshold_px
        self.iou_threshold = iou_threshold or config.collision_iou_threshold
        self.candidate_score_threshold = candidate_score_threshold

    @staticmethod
    def _is_vehicle(class_name: str) -> bool:
        return class_name in config.vehicle_classes

    def analyze_collisions(
        self,
        tracks: List[TrackedObject],
        motion_metrics: Dict[int, TrackMotionMetrics]
    ) -> Tuple[List[CollisionCandidate], float, List[int]]:
        """
        Analyzes all vehicle pairs for collision evidence.
        Returns (list_of_candidates, max_frame_score, involved_track_ids).
        """
        vehicles = [t for t in tracks if self._is_vehicle(t.class_name)]
        candidates: List[CollisionCandidate] = []
        max_score = 0.0
        all_involved_ids: set = set()

        for i in range(len(vehicles)):
            v1 = vehicles[i]
            m1 = motion_metrics.get(v1.track_id, TrackMotionMetrics(v1.track_id))

            for j in range(i + 1, len(vehicles)):
                v2 = vehicles[j]
                m2 = motion_metrics.get(v2.track_id, TrackMotionMetrics(v2.track_id))

                dist = compute_euclidean_distance(v1.center, v2.center)
                iou = compute_iou(v1.bbox, v2.bbox)

                # Relative velocity
                rel_vx = v1.velocity[0] - v2.velocity[0]
                rel_vy = v1.velocity[1] - v2.velocity[1]
                rel_speed = compute_vector_magnitude((rel_vx, rel_vy))

                reasons = []

                # Proximity score (0.0 to 1.0)
                prox_score = max(0.0, 1.0 - (dist / self.proximity_threshold))
                
                # IoU score (0.0 to 1.0)
                iou_score = min(1.0, iou * 4.0)

                # Motion Anomaly score
                motion_anomaly = 0.0
                if m1.is_sudden_deceleration or m2.is_sudden_deceleration:
                    motion_anomaly += 0.3
                    reasons.append("Sudden deceleration")
                if m1.is_abrupt_turn or m2.is_abrupt_turn:
                    motion_anomaly += 0.2
                    reasons.append("Abrupt directional change")
                if m1.is_post_collision_stopped or m2.is_post_collision_stopped:
                    motion_anomaly += 0.3
                    reasons.append("Post-interaction sudden stop")

                if iou >= self.iou_threshold:
                    reasons.append(f"Bounding box overlap (IoU: {iou:.2f})")
                if dist < (self.proximity_threshold * 0.5):
                    reasons.append("High spatial proximity")

                # Suppress false positives for parallel normal passing vehicles
                is_parallel_passing = (
                    iou < 0.10 and
                    not m1.is_sudden_deceleration and
                    not m2.is_sudden_deceleration and
                    not m1.is_abrupt_turn and
                    not m2.is_abrupt_turn and
                    v1.speed > 2.0 and v2.speed > 2.0
                )

                if is_parallel_passing:
                    pair_score = 0.05
                else:
                    pair_score = (prox_score * 0.3) + (iou_score * 0.4) + (motion_anomaly * 0.3)

                if pair_score > max_score:
                    max_score = pair_score

                if pair_score >= self.candidate_score_threshold:
                    candidate = CollisionCandidate(
                        v1=v1,
                        v2=v2,
                        score=pair_score,
                        iou=iou,
                        distance=dist,
                        reasons=reasons
                    )
                    candidates.append(candidate)
                    all_involved_ids.add(v1.track_id)
                    all_involved_ids.add(v2.track_id)

        return candidates, max_score, list(all_involved_ids)
