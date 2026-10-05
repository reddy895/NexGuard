"""
NexGuard — Severity Classification Engine
==========================================
Classifies confirmed accident events into severity levels using a
transparent, rule-based scoring system.

NOTE: This is a PROTOTYPE heuristic system. A trained severity
classifier can replace this module while keeping the interface intact.

Severity levels:
    LOW      — minor incident, limited impact
    MEDIUM   — moderate incident, possible injuries
    HIGH     — serious incident, probable injuries
    CRITICAL — major incident, life-threatening conditions
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from config import cfg
from nexguard.models.schemas import (
    AccidentEvent,
    PersonInvolvement,
    Severity,
    SeverityResult,
    Track,
)
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.accident.severity")


class SeverityClassifier:
    """
    Scores a confirmed accident event and classifies it into a severity level.

    The scoring system is transparent — each signal contributes a documented
    fraction to the final score, making it auditable and adjustable.
    """

    # Signal weights (must sum to ≤ 1.0)
    _W_COLLISION_INTENSITY = 0.25   # IoU + approach velocity combined
    _W_SPEED_CHANGE = 0.20          # Magnitude of speed drop
    _W_MULTI_VEHICLE = 0.15         # Number of vehicles involved
    _W_PERSON_PROXIMITY = 0.20      # Persons near collision zone
    _W_PERSISTENCE = 0.10           # How long the event has been active
    _W_SCENE_DENSITY = 0.10         # Total number of detected objects

    def classify(
        self,
        event: AccidentEvent,
        involved_tracks: List[Track],
        all_tracks: List[Track],
        person_involvements: Optional[List[PersonInvolvement]] = None,
        persistence_frames: int = 0,
    ) -> SeverityResult:
        """
        Classifies the severity of a confirmed accident event.

        Args:
            event:               The confirmed AccidentEvent.
            involved_tracks:     Track objects directly involved in the event.
            all_tracks:          All currently active tracks in the scene.
            person_involvements: Person involvement records (if computed).
            persistence_frames:  How many frames the event has been active.

        Returns:
            SeverityResult with severity label, score, and reasons.
        """
        score = 0.0
        reasons: List[str] = []

        # 1. Collision intensity (from event confidence)
        collision_score = event.confidence * self._W_COLLISION_INTENSITY
        score += collision_score
        if event.confidence > 0.5:
            reasons.append(
                f"high collision confidence ({event.confidence:.0%})"
            )

        # 2. Speed change magnitude
        max_drop = 0.0
        for t in involved_tracks:
            drop = t.speed_drop_ratio
            if drop > max_drop:
                max_drop = drop
        speed_score = max_drop * self._W_SPEED_CHANGE
        score += speed_score
        if max_drop > 0.3:
            reasons.append(f"significant speed reduction ({max_drop:.0%})")

        # 3. Number of vehicles involved
        vehicles_involved = sum(1 for t in involved_tracks if t.is_vehicle)
        vehicle_score = min(1.0, vehicles_involved / 3.0) * self._W_MULTI_VEHICLE
        score += vehicle_score
        if vehicles_involved > 1:
            reasons.append(f"{vehicles_involved} vehicles involved")

        # 4. Person proximity to collision zone
        person_score = 0.0
        if person_involvements:
            high_risk = sum(
                1 for p in person_involvements if p.involvement_level == "HIGH"
            )
            medium_risk = sum(
                1 for p in person_involvements if p.involvement_level == "MEDIUM"
            )
            if high_risk > 0:
                person_score = 1.0
                reasons.append(
                    f"{high_risk} person(s) at HIGH risk near collision"
                )
            elif medium_risk > 0:
                person_score = 0.6
                reasons.append(
                    f"{medium_risk} person(s) at MEDIUM risk near collision"
                )
        score += person_score * self._W_PERSON_PROXIMITY

        # 5. Persistence (longer-lasting events are more severe)
        persistence_score = min(1.0, persistence_frames / 30.0)
        score += persistence_score * self._W_PERSISTENCE
        if persistence_frames > 15:
            reasons.append(f"event persisted for {persistence_frames} frames")

        # 6. Scene density
        total_objects = len(all_tracks)
        density_score = min(1.0, total_objects / 10.0)
        score += density_score * self._W_SCENE_DENSITY
        if total_objects > 5:
            reasons.append(f"high scene density ({total_objects} objects)")

        score = min(1.0, max(0.0, score))

        # Classify into severity bucket
        severity = self._score_to_severity(score)

        log.info(
            f"Severity: {severity.value}  score={score:.3f}  reasons={reasons}"
        )

        return SeverityResult(severity=severity, score=score, reasons=reasons)

    def _score_to_severity(self, score: float) -> Severity:
        if score >= cfg.severity_critical:
            return Severity.CRITICAL
        if score >= cfg.severity_high:
            return Severity.HIGH
        if score >= cfg.severity_medium:
            return Severity.MEDIUM
        return Severity.LOW


# ---------------------------------------------------------------------------
# Person involvement analysis
# ---------------------------------------------------------------------------

def analyze_person_involvement(
    persons: List[Track],
    event: AccidentEvent,
    collision_radius_px: float = 200.0,
) -> List[PersonInvolvement]:
    """
    Analyzes whether detected persons may be involved in an accident event.

    Based purely on observable computer-vision signals:
    - Distance from person to accident zone center
    - Person movement state (stationary vs moving)

    IMPORTANT: This does NOT diagnose injuries. It only reports what
    the camera can observe.

    Args:
        persons:             Active person Track objects.
        event:               The confirmed AccidentEvent.
        collision_radius_px: Distance threshold for involvement assessment.

    Returns:
        List of PersonInvolvement records.
    """
    involvements: List[PersonInvolvement] = []

    cx = event.location.get("cx", 0.0)
    cy = event.location.get("cy", 0.0)

    for person in persons:
        dist = math.hypot(person.cx - cx, person.cy - cy)

        # Estimate movement state
        if person.speed < 0.5:
            movement = "stationary"
        elif person.speed_drop_ratio > 0.6:
            movement = "decelerating"
        else:
            movement = "moving"

        # Involvement level based on distance
        if dist < collision_radius_px * 0.4:
            involvement = "HIGH"
        elif dist < collision_radius_px * 0.8:
            involvement = "MEDIUM"
        elif dist < collision_radius_px:
            involvement = "LOW"
        else:
            continue  # Outside involvement radius — skip

        involvements.append(
            PersonInvolvement(
                track_id=person.track_id,
                distance_px=dist,
                movement=movement,
                involvement_level=involvement,
                last_known_position=(person.cx, person.cy),
            )
        )

    return involvements
