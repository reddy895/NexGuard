"""
NexGuard — Accident Detection Engine
======================================
Implements temporal, multi-signal accident analysis.

IMPORTANT DISCLAIMERS:
  - This module implements a HEURISTIC prototype system.
  - It has NOT been trained or validated on a labelled accident dataset.
  - Accuracy is not guaranteed. It must NOT replace certified safety systems.
  - All detections should be reviewed by qualified personnel.

Design:
  The detector maintains a sliding window of recent track states.
  On each frame it evaluates multiple spatial and kinematic signals
  and produces a collision confidence score. If that score exceeds
  the configured threshold for a sufficient number of consecutive frames,
  an AccidentEvent is emitted.

Signals evaluated:
  1. Bounding-box IoU overlap between vehicle pairs
  2. Centroid proximity between vehicle pairs
  3. Approaching velocity (closing speed) between pairs
  4. Sudden deceleration (speed drop ratio) on involved vehicles
  5. Heading divergence after contact
  6. Pedestrian proximity to collision zone
  7. Single-vehicle sudden stop (possible fixed-object collision)
  8. Direct accident-class detections from a trained model
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from config import cfg
from nexguard.models.schemas import (
    AccidentEvent,
    DetectionMode,
    EventType,
    Track,
)
from nexguard.utils.logging import get_logger
from nexguard.utils.utils import bbox_iou, bbox_centroid_dist, bbox_diagonal, heading_diff, now_iso

log = get_logger("nexguard.accident_detector")


# ---------------------------------------------------------------------------
# Kinematic helpers
# ---------------------------------------------------------------------------

def _approach_velocity(t1: Track, t2: Track) -> float:
    """
    Estimates how fast two objects are closing (positive = approaching).
    Returns 0 if velocity data is unavailable or non-finite.
    """
    try:
        dx, dy = t2.cx - t1.cx, t2.cy - t1.cy
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return 0.0
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return 0.0
        ux, uy = dx / dist, dy / dist
        v1x, v1y = t1.velocity if t1.velocity else (0.0, 0.0)
        v2x, v2y = t2.velocity if t2.velocity else (0.0, 0.0)
        if not all(math.isfinite(v) for v in (v1x, v1y, v2x, v2y)):
            return 0.0
        closing = (v1x * ux + v1y * uy) + (-(v2x * ux + v2y * uy))
        return closing if math.isfinite(closing) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Accident Detector
# ---------------------------------------------------------------------------

class AccidentDetector:
    """
    Temporal accident analysis engine.

    Evaluates motion and spatial signals across a sliding window of frames
    and produces AccidentEvent objects when collision conditions are met.

    A trained model can replace the _score_pair / _score_sudden_stop logic
    while keeping this class's update() interface unchanged.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        confirm_frames: Optional[int] = None,
        cooldown_frames: Optional[int] = None,
    ) -> None:
        self._window = window or cfg.accident_window
        self._threshold = threshold or cfg.accident_threshold
        self._confirm_frames = confirm_frames or cfg.accident_confirm_frames
        self._cooldown_frames = cooldown_frames or cfg.accident_cooldown_frames

        self._frame_buffer: Deque[List[Track]] = deque(maxlen=self._window)
        self._conf_history: Deque[float] = deque(maxlen=self._window)

        self._consecutive: int = 0          # consecutive frames above threshold
        self._last_emit_frame: int = -9999  # frame index of last emitted event
        self._frame_index: int = 0

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def smoothed_confidence(self) -> float:
        """Moving-average confidence over the sliding window."""
        if not self._conf_history:
            return 0.0
        return sum(self._conf_history) / len(self._conf_history)

    @property
    def consecutive_count(self) -> int:
        return self._consecutive

    def update(self, tracks: List[Track]) -> Optional[AccidentEvent]:
        """
        Analyses the current frame's tracks for accident signals.

        Args:
            tracks: Active Track objects from the tracker for this frame.

        Returns:
            AccidentEvent if a confirmed accident is detected, else None.
        """
        self._frame_index += 1
        self._frame_buffer.append(tracks)

        # Need at least 2 frames of history to compute motion
        if len(self._frame_buffer) < 2:
            self._conf_history.append(0.0)
            return None

        accident_tracks = [t for t in tracks if t.is_accident]
        vehicles = [t for t in tracks if t.is_vehicle and t.age >= 3]
        persons = [t for t in tracks if t.is_person]

        event = self._evaluate(vehicles, persons, accident_tracks)

        instant_conf = event.confidence if event else 0.0
        self._conf_history.append(instant_conf)
        smoothed = self.smoothed_confidence

        if event and (instant_conf >= self._threshold or smoothed >= self._threshold * 0.85):
            self._consecutive += 1
            # Blend instant + smoothed confidence
            event.confidence = min(1.0, instant_conf * 0.70 + smoothed * 0.30)
        else:
            self._consecutive = max(0, self._consecutive - 1)

        # Emit only after sustained confirmation and respecting cooldown
        cooldown_elapsed = (self._frame_index - self._last_emit_frame) > self._cooldown_frames
        if event and self._consecutive >= self._confirm_frames and cooldown_elapsed:
            self._last_emit_frame = self._frame_index
            self._consecutive = 0
            log.warning(
                "ACCIDENT CONFIRMED — conf=%.2f smoothed=%.2f type=%s mode=%s tracks=%s",
                event.confidence, smoothed,
                event.event_type.value, event.detection_mode.value, event.track_ids,
            )
            return event

        return None

    def reset(self) -> None:
        """Resets detector state (call when starting a new video)."""
        self._frame_buffer.clear()
        self._conf_history.clear()
        self._consecutive = 0
        self._last_emit_frame = -9999
        self._frame_index = 0

    # ── Internal evaluation ────────────────────────────────────────────────

    def _evaluate(
        self,
        vehicles: List[Track],
        persons: List[Track],
        accident_tracks: List[Track],
    ) -> Optional[AccidentEvent]:
        """
        Evaluates direct model detections first; falls back to kinematics.

        Returns the highest-confidence AccidentEvent found, or None.
        """
        # 1. Direct accident-class detections (trained model)
        if accident_tracks:
            ev = self._evaluate_model_tracks(accident_tracks)
            if ev:
                return ev

        best_event: Optional[AccidentEvent] = None
        best_conf: float = 0.0

        # 2. Vehicle-vehicle kinematic analysis
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                t1, t2 = vehicles[i], vehicles[j]
                conf, signals, etype = self._score_pair(t1, t2, persons)
                if conf > best_conf:
                    best_conf = conf
                    best_event = self._make_event(conf, [t1.track_id, t2.track_id],
                                                  t1, t2, signals, etype)

        # 3. Single-vehicle sudden stop (possible fixed-object collision)
        for t in vehicles:
            conf, signals = self._score_sudden_stop(t)
            if conf > best_conf:
                best_conf = conf
                best_event = AccidentEvent(
                    accident_detected=conf >= self._threshold,
                    confidence=conf,
                    timestamp=now_iso(),
                    track_ids=[t.track_id],
                    location={"cx": round(t.cx, 1), "cy": round(t.cy, 1)},
                    event_type=EventType.SUDDEN_STOP if conf >= self._threshold
                               else EventType.POSSIBLE_ACCIDENT,
                    signals=signals,
                    frame_index=self._frame_index,
                    detection_mode=DetectionMode.HEURISTIC,
                )

        return best_event

    def _make_event(
        self,
        conf: float,
        track_ids: List[int],
        t1: Track,
        t2: Track,
        signals: List[str],
        etype: EventType,
    ) -> AccidentEvent:
        cx = (t1.cx + t2.cx) / 2
        cy = (t1.cy + t2.cy) / 2
        return AccidentEvent(
            accident_detected=conf >= self._threshold,
            confidence=conf,
            timestamp=now_iso(),
            track_ids=track_ids,
            location={"cx": round(cx, 1), "cy": round(cy, 1)},
            event_type=etype if conf >= self._threshold else EventType.POSSIBLE_ACCIDENT,
            signals=signals,
            frame_index=self._frame_index,
            detection_mode=DetectionMode.HEURISTIC,
        )

    def _evaluate_model_tracks(self, accident_tracks: List[Track]) -> Optional[AccidentEvent]:
        """Processes direct accident-class detections from a trained model."""
        best_track, best_conf = None, 0.0
        for t in accident_tracks:
            stability = min(t.hits / max(t.age, 1), 1.0)
            conf = min(1.0, t.confidence * 0.80 + stability * 0.20)
            if conf > best_conf:
                best_conf = conf
                best_track = t

        if best_track is None:
            return None

        return AccidentEvent(
            accident_detected=best_conf >= self._threshold,
            confidence=best_conf,
            timestamp=now_iso(),
            track_ids=[best_track.track_id],
            location={"cx": round(best_track.cx, 1), "cy": round(best_track.cy, 1)},
            event_type=EventType.CONFIRMED_MODEL_EVENT,
            signals=[
                f"trained_model cls={best_track.class_name} "
                f"conf={best_track.confidence:.2f} hits={best_track.hits}/{best_track.age}"
            ],
            frame_index=self._frame_index,
            detection_mode=DetectionMode.TRAINED_MODEL,
            is_model_confirmed=True,
        )

    def _score_pair(
        self,
        t1: Track,
        t2: Track,
        persons: List[Track],
    ) -> Tuple[float, List[str], EventType]:
        """
        Scores a vehicle pair for collision likelihood.

        Returns:
            (confidence_score, list_of_evidence_signals, event_type)
        """
        score: float = 0.0
        signals: List[str] = []
        etype = EventType.VEHICLE_COLLISION

        # 1. Bounding-box IoU overlap
        iou_val = bbox_iou(t1.bbox, t2.bbox)
        if iou_val >= cfg.collision_iou_threshold:
            contribution = min(1.0, iou_val * 4.0)
            score += 0.30 * contribution
            signals.append(f"bbox_overlap={iou_val:.2f} tracks #{t1.track_id}×#{t2.track_id}")

        # 2. Centroid proximity (adaptive threshold based on object scale)
        dist = bbox_centroid_dist(t1.bbox, t2.bbox)
        diag1 = bbox_diagonal(t1.bbox)
        diag2 = bbox_diagonal(t2.bbox)
        adaptive_thresh = max(cfg.proximity_threshold_px, (diag1 + diag2) * 0.45)
        if dist < adaptive_thresh:
            contribution = max(0.0, 1.0 - dist / adaptive_thresh)
            score += 0.20 * contribution
            signals.append(f"proximity={dist:.0f}px (thresh={adaptive_thresh:.0f}px)")

        # 3. Approach (closing) velocity
        approach = _approach_velocity(t1, t2)
        if approach > cfg.approach_velocity_threshold:
            contribution = min(1.0, approach / (cfg.approach_velocity_threshold * 4))
            score += 0.20 * contribution
            signals.append(f"approach_vel={approach:.1f}px/frame")

        # 4. Sudden deceleration on either/both vehicles
        decel_count = 0
        for t in (t1, t2):
            drop = t.speed_drop_ratio
            if drop >= cfg.speed_drop_threshold:
                score += 0.15
                decel_count += 1
                signals.append(f"speed_drop={drop:.0%} track #{t.track_id}")
        if decel_count == 2:
            score += 0.10
            signals.append("mutual_decel: both vehicles decelerating simultaneously")

        # 5. Heading divergence after contact
        if t1.speed > 0.5 and t2.speed > 0.5:
            hdiff = heading_diff(t1.heading, t2.heading)
            if hdiff > cfg.direction_change_threshold:
                contribution = min(1.0, (hdiff - cfg.direction_change_threshold) / 90.0)
                score += 0.15 * contribution
                signals.append(f"heading_divergence={hdiff:.0f}°")
        elif (t1.speed < 0.5 or t2.speed < 0.5) and iou_val >= cfg.collision_iou_threshold:
            score += 0.10
            signals.append("post_impact_standstill during overlap")

        # 6. Person near collision zone
        if persons:
            cx_pair = (t1.cx + t2.cx) / 2
            cy_pair = (t1.cy + t2.cy) / 2
            ped_thresh = adaptive_thresh * 1.5
            for p in persons:
                pd = math.hypot(p.cx - cx_pair, p.cy - cy_pair)
                if pd < ped_thresh:
                    score += 0.10
                    etype = EventType.VEHICLE_PEDESTRIAN
                    signals.append(
                        f"person #{p.track_id} near zone "
                        f"(dist={pd:.0f}px thresh={ped_thresh:.0f}px)"
                    )
                    break

        return min(1.0, score), signals, etype

    def _score_sudden_stop(self, t: Track) -> Tuple[float, List[str]]:
        """
        Scores a single vehicle for a sudden-stop event
        (possible collision with a stationary object or barrier).
        """
        if len(t.speed_history) < 5:
            return 0.0, []

        score: float = 0.0
        signals: List[str] = []

        drop = t.speed_drop_ratio
        if drop >= cfg.speed_drop_threshold * 1.2:
            score += 0.40
            signals.append(f"sudden_stop speed_drop={drop:.0%} track #{t.track_id}")

        max_recent = max(t.speed_history[:-2], default=0.0)
        if max_recent > 5.0 and t.speed < 1.0:
            score += 0.30
            signals.append(
                f"high_speed_to_stationary: was {max_recent:.1f}px/f now {t.speed:.1f}px/f"
            )

        return min(1.0, score), signals
