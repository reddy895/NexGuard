"""
NexGuard — Severity Classification Engine
==========================================
Classifies confirmed accident events into severity levels using a
transparent, rule-based scoring system.

NOTE: This is a PROTOTYPE heuristic. A trained severity classifier
can replace this module while keeping the same interface.

Severity levels (per specification):
    NO_ACCIDENT — baseline state, no event
    POSSIBLE    — potential collision under evaluation
    LOW         — low-impact contact, no reliable person-fall evidence
    MEDIUM      — person-fall evidence or significant secondary impact
    HIGH        — major collision, rollover, or multiple strong signals
    UNCERTAIN   — insufficient or conflicting evidence

Blood / injury detection is NOT implemented (no validated model available).
UNCERTAIN is used when evidence is insufficient for a reliable classification.
"""

from __future__ import annotations

import math
from typing import List, Optional

from config import cfg
from nexguard.models.schemas import (
    AccidentEvent,
    DetectionMode,
    PersonInvolvement,
    Severity,
    SeverityResult,
    Track,
)
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.severity")


class SeverityClassifier:
    """
    Scores a confirmed accident event and classifies it into a severity level.

    Signal weights (must sum to ≤ 1.0):
        _W_COLLISION   — IoU + approach velocity combined   0.25
        _W_SPEED       — magnitude of speed drop             0.20
        _W_VEHICLES    — number of vehicles involved         0.15
        _W_PERSON      — persons near collision zone         0.20
        _W_PERSISTENCE — how long the event has been active  0.10
        _W_DENSITY     — total detected objects in scene     0.10

    The scoring is transparent and auditable: each signal's contribution
    is documented and independently adjustable.
    """

    _W_COLLISION   = 0.25
    _W_SPEED       = 0.20
    _W_VEHICLES    = 0.15
    _W_PERSON      = 0.20
    _W_PERSISTENCE = 0.10
    _W_DENSITY     = 0.10

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
            involved_tracks:     Tracks directly involved in the event.
            all_tracks:          All active tracks in the current frame.
            person_involvements: Person involvement records (optional).
            persistence_frames:  Consecutive frames the event has been active.

        Returns:
            SeverityResult with severity label, score, and evidence reasons.
        """
        score: float = 0.0
        reasons: List[str] = []

        # 1. Collision intensity (from event confidence score)
        col_score = event.confidence * self._W_COLLISION
        score += col_score
        if event.confidence > 0.50:
            reasons.append(f"high collision confidence ({event.confidence:.0%})")

        # 2. Speed change magnitude across involved tracks
        max_drop = max((t.speed_drop_ratio for t in involved_tracks), default=0.0)
        score += max_drop * self._W_SPEED
        if max_drop > 0.30:
            reasons.append(f"significant speed reduction ({max_drop:.0%})")

        # 3. Number of vehicles involved
        n_vehicles = sum(1 for t in involved_tracks if t.is_vehicle)
        score += min(1.0, n_vehicles / 3.0) * self._W_VEHICLES
        if n_vehicles > 1:
            reasons.append(f"{n_vehicles} vehicles involved")

        # 4. Person proximity to collision zone
        person_score = 0.0
        if person_involvements:
            high_risk = sum(1 for p in person_involvements if p.involvement_level == "HIGH")
            med_risk  = sum(1 for p in person_involvements if p.involvement_level == "MEDIUM")
            if high_risk > 0:
                person_score = 1.0
                reasons.append(f"{high_risk} person(s) at HIGH-risk near collision zone")
            elif med_risk > 0:
                person_score = 0.60
                reasons.append(f"{med_risk} person(s) at MEDIUM-risk near collision zone")
        score += person_score * self._W_PERSON

        # 5. Event persistence (longer events correlate with greater severity)
        persistence_score = min(1.0, persistence_frames / 30.0)
        score += persistence_score * self._W_PERSISTENCE
        if persistence_frames > 15:
            reasons.append(f"event persisted {persistence_frames} frames")

        # 6. Scene density
        density_score = min(1.0, len(all_tracks) / 10.0)
        score += density_score * self._W_DENSITY
        if len(all_tracks) > 5:
            reasons.append(f"dense scene ({len(all_tracks)} objects)")

        score = min(1.0, max(0.0, score))

        # UNCERTAIN path: trained accident model confirmed but confidence is
        # borderline — not enough evidence to commit to LOW/MEDIUM/HIGH
        if (
            event.detection_mode == DetectionMode.TRAINED_MODEL
            and event.is_model_confirmed
            and score < cfg.severity_low
        ):
            log.info("Severity: UNCERTAIN (trained model, borderline confidence)")
            return SeverityResult(
                severity=Severity.UNCERTAIN,
                score=score,
                reasons=["trained model detection with insufficient supporting signals"],
            )

        severity = self._score_to_severity(score)
        log.info("Severity: %s  score=%.3f  reasons=%s", severity.value, score, reasons)
        return SeverityResult(severity=severity, score=score, reasons=reasons)

    def _score_to_severity(self, score: float) -> Severity:
        if score >= cfg.severity_high:
            return Severity.HIGH
        if score >= cfg.severity_medium:
            return Severity.MEDIUM
        if score >= cfg.severity_low:
            return Severity.LOW
        return Severity.LOW  # anything above accident threshold is at least LOW


# ---------------------------------------------------------------------------
# Person involvement analysis
# ---------------------------------------------------------------------------

def analyze_person_involvement(
    persons: List[Track],
    event: AccidentEvent,
    collision_radius_px: float = 200.0,
) -> List[PersonInvolvement]:
    """
    Analyses whether detected persons may be involved in the accident zone.

    Based purely on observable CV signals:
    - Distance from person centroid to accident zone centre
    - Person movement state (stationary vs moving vs decelerating)

    IMPORTANT: This does NOT diagnose injuries.
    A stationary person near a collision zone is not necessarily injured.
    Single-frame positional evidence is insufficient to classify a fall.

    Args:
        persons:             Active person Track objects.
        event:               The confirmed AccidentEvent.
        collision_radius_px: Distance threshold for involvement assessment.

    Returns:
        List of PersonInvolvement records for persons within the radius.
    """
    involvements: List[PersonInvolvement] = []
    cx = event.location.get("cx", 0.0)
    cy = event.location.get("cy", 0.0)

    for person in persons:
        dist = math.hypot(person.cx - cx, person.cy - cy)

        if person.speed < 0.5:
            movement = "stationary"
        elif person.speed_drop_ratio > 0.60:
            movement = "decelerating"
        else:
            movement = "moving"

        if dist < collision_radius_px * 0.40:
            level = "HIGH"
        elif dist < collision_radius_px * 0.80:
            level = "MEDIUM"
        elif dist < collision_radius_px:
            level = "LOW"
        else:
            continue

        involvements.append(PersonInvolvement(
            track_id=person.track_id,
            distance_px=dist,
            movement=movement,
            involvement_level=level,
            position=(person.cx, person.cy),
        ))

    return involvements
