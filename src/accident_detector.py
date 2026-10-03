"""
NexGuard Temporal Accident Detection Engine
Manages temporal state machine (NORMAL -> SUSPECTED_COLLISION -> CONFIRMING -> ACCIDENT_CONFIRMED -> RECOVERY),
false-positive suppression, multi-frame evidence verification, person proximity assessment,
and Layer 2 custom accident model evidence fusion.
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from config import config
from utils.geometry import compute_euclidean_distance
from src.tracker import TrackedObject
from src.motion_analyzer import MotionAnalyzer, TrackMotionMetrics
from src.collision_analyzer import CollisionAnalyzer, CollisionCandidate


class AccidentState(Enum):
    NORMAL = "NORMAL"
    SUSPECTED_COLLISION = "SUSPECTED_COLLISION"
    CONFIRMING = "CONFIRMING"
    ACCIDENT_CONFIRMED = "ACCIDENT_CONFIRMED"
    RECOVERY = "RECOVERY"


class AccidentAnalysisResult:
    """Encapsulates output of accident analysis engine for a single frame."""

    def __init__(
        self,
        is_accident: bool,
        state: AccidentState,
        confidence_score: float,
        involved_track_ids: List[int],
        people_near_accident: int,
        reasons: List[str],
        candidates: List[CollisionCandidate],
        temporal_frames: int
    ):
        self.is_accident: bool = is_accident
        self.state: AccidentState = state
        self.confidence_score: float = round(confidence_score, 4)
        self.involved_track_ids: List[int] = involved_track_ids
        self.people_near_accident: int = people_near_accident
        self.reasons: List[str] = reasons
        self.candidates: List[CollisionCandidate] = candidates
        self.temporal_frames: int = temporal_frames

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_accident": self.is_accident,
            "state": self.state.value,
            "confidence_score": self.confidence_score,
            "involved_track_ids": self.involved_track_ids,
            "people_near_accident": self.people_near_accident,
            "reasons": self.reasons,
            "temporal_frames": self.temporal_frames
        }


class TemporalAccidentDetector:
    """Multi-frame temporal accident analysis engine with state machine and state decay."""

    def __init__(
        self,
        confirmation_frames: int = None,
        candidate_threshold: float = 0.35
    ):
        self.confirmation_frames = confirmation_frames or config.accident_confirmation_frames
        self.candidate_threshold = candidate_threshold
        
        self.motion_analyzer = MotionAnalyzer(
            speed_drop_threshold=config.sudden_speed_drop_threshold,
            direction_change_threshold=config.direction_change_threshold
        )
        self.collision_analyzer = CollisionAnalyzer(
            proximity_threshold=config.proximity_threshold_px,
            iou_threshold=config.collision_iou_threshold,
            candidate_score_threshold=candidate_threshold
        )

        self.state: AccidentState = AccidentState.NORMAL
        self.evidence_counter: int = 0
        self.recovery_counter: int = 0
        self.active_involved_tracks: List[int] = []

    def reset(self):
        """Resets temporal state machine."""
        self.state = AccidentState.NORMAL
        self.evidence_counter = 0
        self.recovery_counter = 0
        self.active_involved_tracks.clear()

    def process_frame(
        self,
        tracks: List[TrackedObject],
        custom_model_accident_detected: bool = False
    ) -> AccidentAnalysisResult:
        """
        Processes frame tracks and updates temporal state machine.
        Fuses Layer 1 temporal evidence with Layer 2 custom accident model detection.
        """
        # Step 1: Motion Analysis
        motion_metrics = self.motion_analyzer.analyze_all_tracks(tracks)

        # Step 2: Collision Analysis
        candidates, max_score, involved_ids = self.collision_analyzer.analyze_collisions(tracks, motion_metrics)

        # Fuse Layer 2 Custom Model Evidence
        if custom_model_accident_detected:
            max_score = max(max_score, 0.75)

        reasons = []
        for c in candidates:
            reasons.extend(c.reasons)
        if custom_model_accident_detected:
            reasons.append("Custom accident YOLO model positive detection")

        # Step 3: State Machine Transitions
        if max_score >= self.candidate_threshold:
            self.evidence_counter += 1
            self.recovery_counter = 0
            if involved_ids:
                self.active_involved_tracks = list(set(self.active_involved_tracks + involved_ids))
        else:
            self.evidence_counter = max(0, self.evidence_counter - 1)
            self.recovery_counter += 1

        # State transition logic
        if self.evidence_counter == 0:
            self.state = AccidentState.NORMAL
            self.active_involved_tracks.clear()
        elif self.evidence_counter == 1:
            self.state = AccidentState.SUSPECTED_COLLISION
        elif 1 < self.evidence_counter < self.confirmation_frames:
            self.state = AccidentState.CONFIRMING
        elif self.evidence_counter >= self.confirmation_frames:
            self.state = AccidentState.ACCIDENT_CONFIRMED

        is_accident = (self.state == AccidentState.ACCIDENT_CONFIRMED)

        # Step 4: Person Involvement Proximity Check
        people_near = 0
        if self.active_involved_tracks:
            involved_objects = [t for t in tracks if t.track_id in self.active_involved_tracks]
            person_objects = [t for t in tracks if t.class_name == "person"]

            for p in person_objects:
                for inv in involved_objects:
                    d = compute_euclidean_distance(p.center, inv.center)
                    if d < (config.proximity_threshold_px * 1.5):
                        people_near += 1
                        break

        return AccidentAnalysisResult(
            is_accident=is_accident,
            state=self.state,
            confidence_score=max_score,
            involved_track_ids=self.active_involved_tracks if is_accident else [],
            people_near_accident=people_near,
            reasons=list(set(reasons)),
            candidates=candidates,
            temporal_frames=self.evidence_counter
        )
