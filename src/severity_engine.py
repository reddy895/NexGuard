"""
NexGuard Accident Severity Classification Engine
Evaluates measurable spatial, vehicle count, and pedestrian proximity evidence
to classify accident severity into NORMAL, LOW, MEDIUM, HIGH, or CRITICAL.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class SeverityEvaluation:
    level: str
    score: float
    confidence: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons
        }


class SeverityEngine:
    """Classifies accident incident severity based on empirical evidence parameters."""

    @staticmethod
    def classify(
        collision_detected: bool,
        collision_score: float = 0.0,
        vehicles_involved: int = 0,
        people_involved: int = 0,
        has_vulnerable_user: bool = False,
        post_stop: bool = False
    ) -> SeverityEvaluation:
        """
        Classifies severity using measurable evidence:
        - NORMAL: No accident confirmed
        - LOW: Collision score < 0.35 or 1 vehicle interaction
        - MEDIUM: 2 vehicles involved, no pedestrians
        - HIGH: Collision with pedestrian nearby OR vulnerable road user (motorcycle/bicycle)
        - CRITICAL: >= 3 vehicles OR multi-pedestrian involvement
        """
        if not collision_detected:
            return SeverityEvaluation(
                level="NORMAL",
                score=0.0,
                confidence=1.0,
                reasons=["No active accident confirmed"]
            )

        reasons = []

        if vehicles_involved >= 3:
            reasons.append(f"Multiple vehicles involved ({vehicles_involved})")
            if people_involved >= 1:
                reasons.append(f"Pedestrian proximity detected ({people_involved})")
                return SeverityEvaluation(level="CRITICAL", score=0.95, confidence=0.90, reasons=reasons)
            return SeverityEvaluation(level="CRITICAL", score=0.85, confidence=0.85, reasons=reasons)

        if people_involved >= 1:
            reasons.append(f"Pedestrian in collision zone ({people_involved})")
            if vehicles_involved >= 2 or has_vulnerable_user:
                return SeverityEvaluation(level="HIGH", score=0.80, confidence=0.85, reasons=reasons)

        if has_vulnerable_user:
            reasons.append("Vulnerable road user (motorcycle/bicycle) involved")
            return SeverityEvaluation(level="HIGH", score=0.75, confidence=0.80, reasons=reasons)

        if vehicles_involved >= 2:
            reasons.append("Two-vehicle collision detected")
            return SeverityEvaluation(level="MEDIUM", score=0.65, confidence=0.75, reasons=reasons)

        reasons.append("Single vehicle anomaly detected")
        return SeverityEvaluation(level="LOW", score=0.40, confidence=0.60, reasons=reasons)
