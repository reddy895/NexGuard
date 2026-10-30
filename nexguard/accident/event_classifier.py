"""
NexGuard — Accident Event Classifier
====================================
Combines temporal features, trajectory kinematics, and model inference
to classify accident events into structured event types:
- Evaluates multi-signal collision evidence
- Distinguishes true impacts from near-misses and normal traffic deceleration
- Returns calibrated probability and forensic evidence tags
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from nexguard.accident.temporal import TemporalAccidentFeatures
from nexguard.models.schemas import EventType

@dataclass
class EventClassificationResult:
    event_type: EventType
    confidence: float
    is_accident: bool
    evidence_signals: List[str] = field(default_factory=list)

class AccidentEventClassifier:
    """
    Classifies multi-frame vehicle interactions into collision events.
    """

    def __init__(
        self,
        collision_thresh: float = 0.55,
        near_miss_thresh: float = 0.35,
    ) -> None:
        self.collision_thresh = collision_thresh
        self.near_miss_thresh = near_miss_thresh

    def classify_interaction(
        self,
        features: TemporalAccidentFeatures,
        has_pedestrian: bool = False,
    ) -> EventClassificationResult:
        score = 0.0
        signals = []

        # 1. Overlap evidence
        if features.max_iou >= 0.08:
            contrib = min(0.30, features.max_iou * 1.5)
            score += contrib
            signals.append(f"iou_overlap={features.max_iou:.2f} ({features.consecutive_overlap_frames}f)")

        # 2. Proximity evidence
        if features.min_dist_px < 120.0:
            proximity_factor = max(0.0, 1.0 - features.min_dist_px / 120.0)
            score += 0.20 * proximity_factor
            signals.append(f"min_distance={features.min_dist_px:.0f}px")

        # 3. Approach velocity
        if features.max_approach_vel > 4.0:
            contrib = min(0.20, features.max_approach_vel / 20.0)
            score += contrib
            signals.append(f"closing_speed={features.max_approach_vel:.1f}px/f")

        # 4. Deceleration & Standstill
        if features.max_deceleration_ratio > 0.40:
            score += 0.15
            signals.append(f"speed_drop={features.max_deceleration_ratio:.0%}")
        if features.post_impact_standstill_frames >= 3:
            score += 0.15
            signals.append(f"post_impact_standstill={features.post_impact_standstill_frames}f")

        # 5. Direct Model Probability Blend
        if features.mean_model_confidence > 0.50:
            score += 0.20 * features.mean_model_confidence
            signals.append(f"model_prior_conf={features.mean_model_confidence:.2f}")

        final_conf = min(1.0, score)

        # Classification decision
        if final_conf >= self.collision_thresh:
            etype = EventType.VEHICLE_PEDESTRIAN if has_pedestrian else EventType.VEHICLE_COLLISION
            return EventClassificationResult(
                event_type=etype,
                confidence=final_conf,
                is_accident=True,
                evidence_signals=signals,
            )
        elif final_conf >= self.near_miss_thresh:
            return EventClassificationResult(
                event_type=EventType.NEAR_MISS,
                confidence=final_conf,
                is_accident=False,
                evidence_signals=signals,
            )
        else:
            return EventClassificationResult(
                event_type=EventType.NO_ACCIDENT,
                confidence=final_conf,
                is_accident=False,
                evidence_signals=signals,
            )
