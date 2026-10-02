"""
NexGuard AI-Derived Incident Severity Classifier
"""

from typing import Dict, Any, List
from pydantic import BaseModel


class SeverityEvaluation(BaseModel):
    level: str  # NORMAL, LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    reasons: List[str]
    score: float


class SeverityClassifier:
    """Classifies AI-derived incident severity based on collision metrics, tracking evidence, and vulnerable road user involvement."""

    @staticmethod
    def classify(
        collision_detected: bool,
        collision_score: float,
        vehicles_involved: int,
        people_involved: int,
        has_vulnerable_user: bool = False,
        post_stop: bool = False
    ) -> SeverityEvaluation:
        reasons = []

        if not collision_detected or collision_score < 0.25:
            return SeverityEvaluation(
                level="NORMAL",
                confidence=round(max(0.0, 1.0 - collision_score), 4),
                reasons=["No abnormal collision patterns or high-speed proximity events detected."],
                score=round(collision_score, 4)
            )

        # Base severity score calculated from physical indicators
        base_score = collision_score

        # Multiplier / boosts based on safety impact
        if vehicles_involved >= 2:
            base_score += 0.15
            reasons.append(f"Multiple vehicles ({vehicles_involved}) involved in collision zone.")
            
        if people_involved > 0:
            base_score += 0.25
            reasons.append(f"Vulnerable pedestrians/persons ({people_involved}) detected inside collision threshold.")

        if has_vulnerable_user:
            base_score += 0.15
            reasons.append("Motorcycle or bicycle collision pattern identified.")

        if post_stop:
            base_score += 0.10
            reasons.append("Abnormal post-interaction vehicle standstill observed.")

        # Cap score at 1.0
        final_score = min(1.0, max(0.0, base_score))

        # Categorize into 5 official levels
        if final_score >= 0.85:
            level = "CRITICAL"
        elif final_score >= 0.65:
            level = "HIGH"
        elif final_score >= 0.45:
            level = "MEDIUM"
        elif final_score >= 0.25:
            level = "LOW"
        else:
            level = "NORMAL"

        if not reasons:
            reasons.append("Proximity/motion dynamics indicate potential low-speed vehicle interaction.")

        return SeverityEvaluation(
            level=level,
            confidence=round(final_score, 4),
            reasons=reasons,
            score=round(final_score, 4)
        )
