"""
NexGuard AI-Derived Incident Severity Engine
"""

from typing import List, Dict, Any


class SeverityEvaluation:
    def __init__(self, level: str, confidence: float, score: float, reasons: List[str]):
        self.level = level            # NORMAL, LOW, MEDIUM, HIGH, CRITICAL
        self.confidence = confidence  # 0.0 to 1.0
        self.score = score            # 0.0 to 1.0
        self.reasons = reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 4),
            "reasons": self.reasons
        }


class SeverityEngine:
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
                score=round(collision_score, 4),
                reasons=["Normal traffic flow. No high-speed vehicle proximity or collision indicators."]
            )

        base_score = collision_score

        if vehicles_involved >= 2:
            base_score += 0.15
            reasons.append(f"Multiple vehicles ({vehicles_involved}) involved in collision proximity zone.")

        if people_involved > 0:
            base_score += 0.25
            reasons.append(f"Pedestrians/Persons ({people_involved}) involved near impact zone.")

        if has_vulnerable_user:
            base_score += 0.15
            reasons.append("Vulnerable road user (Motorcycle/Bicycle) collision pattern detected.")

        if post_stop:
            base_score += 0.10
            reasons.append("Post-collision vehicle standstill observed.")

        final_score = min(1.0, max(0.0, base_score))

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
            reasons.append("Proximity dynamics indicate potential vehicle interaction.")

        return SeverityEvaluation(
            level=level,
            confidence=round(final_score, 4),
            score=round(final_score, 4),
            reasons=reasons
        )
