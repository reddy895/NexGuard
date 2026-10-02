"""
NexGuard Rule-Based Temporal Accident Detection Engine
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from backend.app.config import settings
from backend.app.tracking.tracker import TrackedObject
from backend.app.severity.classifier import SeverityClassifier, SeverityEvaluation


class AccidentAnalysisResult:
    def __init__(
        self,
        is_accident: bool,
        severity: SeverityEvaluation,
        incident_id: Optional[str] = None,
        interacting_tracks: List[int] = None,
        metrics: Dict = None
    ):
        self.is_accident = is_accident
        self.severity = severity
        self.incident_id = incident_id
        self.interacting_tracks = interacting_tracks or []
        self.metrics = metrics or {}

    def to_dict(self) -> Dict:
        return {
            "is_accident": self.is_accident,
            "severity_level": self.severity.level,
            "confidence": self.severity.confidence,
            "score": self.severity.score,
            "reasons": self.severity.reasons,
            "incident_id": self.incident_id,
            "interacting_tracks": self.interacting_tracks,
            "metrics": self.metrics
        }


class AccidentDetector:
    def __init__(self):
        self.temporal_evidence_counter = 0
        self.min_temporal_frames = settings.MIN_TEMPORAL_FRAMES
        self.proximity_threshold = settings.PROXIMITY_THRESHOLD_PX
        self.last_interacting_tracks: List[int] = []

    def reset(self):
        self.temporal_evidence_counter = 0
        self.last_interacting_tracks.clear()

    @staticmethod
    def _compute_iou(boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    @staticmethod
    def _is_vehicle(class_name: str) -> bool:
        return class_name in ["car", "motorcycle", "bus", "truck", "bicycle"]

    def analyze_frame(self, tracks: List[TrackedObject]) -> AccidentAnalysisResult:
        """Analyzes active object tracks for multi-frame accident evidence."""
        vehicles = [t for t in tracks if self._is_vehicle(t.class_name)]
        persons = [t for t in tracks if t.class_name == "person"]

        max_pair_score = 0.0
        best_interacting_ids: List[int] = []
        has_vulnerable = False
        post_stop = False
        num_vehicles_involved = 0

        # Check vehicle-to-vehicle / vehicle-to-vulnerable interactions
        for i in range(len(vehicles)):
            v1 = vehicles[i]
            for j in range(i + 1, len(vehicles)):
                v2 = vehicles[j]

                # Distance between centroids
                dist = math.hypot(v1.center[0] - v2.center[0], v1.center[1] - v2.center[1])
                iou = self._compute_iou(v1.bbox, v2.bbox)

                # Approach vector alignment (dot product of velocity vectors)
                rel_vx = v1.velocity[0] - v2.velocity[0]
                rel_vy = v1.velocity[1] - v2.velocity[1]
                rel_speed = math.hypot(rel_vx, rel_vy)

                # Score factors
                proximity_score = max(0.0, 1.0 - (dist / self.proximity_threshold))
                iou_score = min(1.0, iou * 3.0)  # Boost overlap weight
                speed_score = min(1.0, rel_speed / settings.VELOCITY_APPROACH_THRESHOLD)

                # Pair collision indicator
                pair_score = (proximity_score * 0.4) + (iou_score * 0.4) + (speed_score * 0.2)

                if pair_score > max_pair_score:
                    max_pair_score = pair_score
                    best_interacting_ids = [v1.track_id, v2.track_id]

                if dist < self.proximity_threshold or iou > 0.05:
                    num_vehicles_involved += 1
                    if v1.class_name in ["motorcycle", "bicycle"] or v2.class_name in ["motorcycle", "bicycle"]:
                        has_vulnerable = True

                    # Check for sudden post-collision deceleration/stopping
                    if v1.speed < 0.5 and v2.speed < 0.5 and (v1.frames_tracked > 5 or v2.frames_tracked > 5):
                        post_stop = True

        # Count persons near interaction zone
        people_near_collision = 0
        if best_interacting_ids:
            col_x = np.mean([t.center[0] for t in tracks if t.track_id in best_interacting_ids])
            col_y = np.mean([t.center[1] for t in tracks if t.track_id in best_interacting_ids])

            for p in persons:
                p_dist = math.hypot(p.center[0] - col_x, p.center[1] - col_y)
                if p_dist < (self.proximity_threshold * 1.5):
                    people_near_collision += 1
                    best_interacting_ids.append(p.track_id)

        # Multi-frame temporal accumulation
        if max_pair_score >= 0.35:
            self.temporal_evidence_counter += 1
        else:
            self.temporal_evidence_counter = max(0, self.temporal_evidence_counter - 1)

        is_accident = (self.temporal_evidence_counter >= self.min_temporal_frames) and (max_pair_score >= 0.40)

        # Run severity classifier
        severity = SeverityClassifier.classify(
            collision_detected=is_accident,
            collision_score=max_pair_score if is_accident else 0.0,
            vehicles_involved=max(num_vehicles_involved, 2 if is_accident else 0),
            people_involved=people_near_collision,
            has_vulnerable_user=has_vulnerable,
            post_stop=post_stop
        )

        return AccidentAnalysisResult(
            is_accident=is_accident,
            severity=severity,
            interacting_tracks=best_interacting_ids if is_accident else [],
            metrics={
                "max_collision_score": round(max_pair_score, 4),
                "temporal_frames": self.temporal_evidence_counter,
                "vehicles_involved": num_vehicles_involved,
                "people_involved": people_near_collision
            }
        )
