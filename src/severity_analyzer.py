"""Severity estimation for NexGuard accident events."""

from __future__ import annotations


class SeverityAnalyzer:
    """Maps a normalized collision score into a severity bucket."""

    @staticmethod
    def compute(collision_score: float, vehicles_involved: int, persons_nearby: int) -> str:
        """Return LOW, MEDIUM, or HIGH based on event signal strength."""
        normalized = max(0.0, min(1.0, float(collision_score)))
        vehicle_factor = min(vehicles_involved, 3) * 0.10
        person_factor = min(persons_nearby, 2) * 0.10
        weighted_score = normalized * 0.35 + vehicle_factor + person_factor

        if weighted_score >= 0.70:
            return "HIGH"
        if weighted_score >= 0.45:
            return "MEDIUM"
        return "LOW"
