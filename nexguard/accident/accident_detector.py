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
  On each frame, it evaluates multiple spatial and motion signals
  to produce a collision confidence score. If the score exceeds
  the configured threshold for a sufficient number of consecutive
  frames, an AccidentEvent is emitted.

Signals evaluated:
  1. Bounding-box IoU overlap between vehicle pairs
  2. Centroid proximity between vehicle pairs
  3. Approaching velocity (closing speed)
  4. Sudden deceleration (speed drop ratio)
  5. Sudden heading change (direction change)
  6. Vehicle-to-pedestrian proximity
  7. Person stationary near potential collision zone
"""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from config import cfg
from nexguard.models.schemas import AccidentEvent, EventType, Track
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.accident")


# ---------------------------------------------------------------------------
# Geometry helpers (self-contained — no circular imports)
# ---------------------------------------------------------------------------

def _iou(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _dist(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    return math.hypot(cx2 - cx1, cy2 - cy1)


def _approach_velocity(t1: Track, t2: Track) -> float:
    """
    Estimates how fast two objects are moving toward each other.
    Positive value means they are closing.
    """
    dx = t2.cx - t1.cx
    dy = t2.cy - t1.cy
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    # Unit vector from t1 to t2
    ux, uy = dx / dist, dy / dist
    # Project each velocity onto the closing axis
    v1_closing = t1.velocity[0] * ux + t1.velocity[1] * uy
    v2_closing = -(t2.velocity[0] * ux + t2.velocity[1] * uy)
    return v1_closing + v2_closing


def _heading_diff(h1: float, h2: float) -> float:
    """Smallest angular difference between two headings (0–180°)."""
    diff = abs(h1 - h2) % 360
    return min(diff, 360 - diff)


# ---------------------------------------------------------------------------
# Accident Detector
# ---------------------------------------------------------------------------

class AccidentDetector:
    """
    Temporal accident analysis engine.

    Evaluates motion and spatial signals across a sliding window of frames
    and produces AccidentEvent objects when collision conditions are met.

    The detection is heuristic — it estimates accident probability from
    observable computer-vision signals rather than from a trained classifier.
    A trained model can replace the scoring logic while keeping this interface.
    """

    def __init__(
        self,
        window: Optional[int] = None,
        threshold: Optional[float] = None,
        confirm_frames: Optional[int] = None,
    ) -> None:
        self._window = window or cfg.accident_window
        self._threshold = threshold or cfg.accident_threshold
        self._confirm_frames = confirm_frames or cfg.accident_confirm_frames

        # Sliding window: stores per-frame track snapshots
        self._frame_buffer: Deque[List[Track]] = deque(maxlen=self._window)

        # Confirmation counter — how many consecutive frames exceeded threshold
        self._consecutive_detections: int = 0

        # Track the last emitted event to avoid duplicate emissions
        self._last_event_frame: int = -999

        self._frame_index: int = 0

    def update(self, tracks: List[Track]) -> Optional[AccidentEvent]:
        """
        Analyzes the current frame's tracks for accident signals.

        Args:
            tracks: Active Track objects from the tracker for this frame.

        Returns:
            AccidentEvent if a confirmed accident is detected, else None.
        """
        self._frame_index += 1
        self._frame_buffer.append(tracks)

        # Need at least 2 frames of history to compute motion
        if len(self._frame_buffer) < 2:
            return None

        # Evaluate all vehicle pairs in the current frame
        vehicles = [t for t in tracks if t.is_vehicle and t.age >= 3]
        persons = [t for t in tracks if t.is_person]

        event = self._evaluate(vehicles, persons)

        if event and event.confidence >= self._threshold:
            self._consecutive_detections += 1
        else:
            self._consecutive_detections = max(0, self._consecutive_detections - 1)

        # Only emit if confirmed for enough consecutive frames
        if self._consecutive_detections >= self._confirm_frames:
            if event and (self._frame_index - self._last_event_frame) > self._confirm_frames:
                self._last_event_frame = self._frame_index
                self._consecutive_detections = 0  # Reset after emission
                log.info(
                    f"[ACCIDENT CONFIRMED] confidence={event.confidence:.2f} "
                    f"type={event.event_type.value} "
                    f"tracks={event.track_ids}"
                )
                return event

        return None

    def _evaluate(
        self, vehicles: List[Track], persons: List[Track]
    ) -> Optional[AccidentEvent]:
        """
        Evaluates all vehicle-vehicle pairs and vehicle-person proximity.

        Returns:
            The highest-confidence AccidentEvent found, or None.
        """
        best_event: Optional[AccidentEvent] = None
        best_conf: float = 0.0

        # --- Vehicle-vehicle analysis ---
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                t1, t2 = vehicles[i], vehicles[j]
                conf, signals, etype = self._score_pair(t1, t2, persons)
                if conf > best_conf:
                    best_conf = conf
                    cx = (t1.cx + t2.cx) / 2
                    cy = (t1.cy + t2.cy) / 2
                    best_event = AccidentEvent(
                        accident_detected=conf >= self._threshold,
                        confidence=conf,
                        timestamp=datetime.now().isoformat(),
                        track_ids=[t1.track_id, t2.track_id],
                        location={"cx": round(cx, 1), "cy": round(cy, 1)},
                        event_type=etype,
                        signals=signals,
                        frame_index=self._frame_index,
                    )

        # --- Single-vehicle sudden-stop (potential collision with static object) ---
        for t in vehicles:
            conf, signals = self._score_sudden_stop(t)
            if conf > best_conf:
                best_conf = conf
                best_event = AccidentEvent(
                    accident_detected=conf >= self._threshold,
                    confidence=conf,
                    timestamp=datetime.now().isoformat(),
                    track_ids=[t.track_id],
                    location={"cx": round(t.cx, 1), "cy": round(t.cy, 1)},
                    event_type=EventType.SUDDEN_STOP,
                    signals=signals,
                    frame_index=self._frame_index,
                )

        return best_event

    def _score_pair(
        self,
        t1: Track,
        t2: Track,
        persons: List[Track],
    ) -> Tuple[float, List[str], EventType]:
        """
        Scores a vehicle pair for collision likelihood.

        Returns:
            (confidence_score, list_of_signals, event_type)
        """
        score = 0.0
        signals: List[str] = []
        etype = EventType.VEHICLE_COLLISION

        # 1. Bounding-box overlap (IoU)
        iou_val = _iou(t1.bbox, t2.bbox)
        if iou_val >= cfg.collision_iou_threshold:
            contribution = min(1.0, iou_val * 4.0)  # scale: 0.10 IoU → 0.40
            score += 0.30 * contribution
            signals.append(
                f"bbox_overlap={iou_val:.2f} between tracks #{t1.track_id} and #{t2.track_id}"
            )

        # 2. Centroid proximity
        dist = _dist(t1.bbox, t2.bbox)
        if dist < cfg.proximity_threshold_px:
            contribution = max(0.0, 1.0 - dist / cfg.proximity_threshold_px)
            score += 0.20 * contribution
            signals.append(
                f"proximity={dist:.0f}px (threshold={cfg.proximity_threshold_px:.0f}px)"
            )

        # 3. Approach (closing) velocity
        approach_vel = _approach_velocity(t1, t2)
        if approach_vel > cfg.approach_velocity_threshold:
            contribution = min(1.0, approach_vel / (cfg.approach_velocity_threshold * 4))
            score += 0.20 * contribution
            signals.append(f"approach_velocity={approach_vel:.1f}px/frame")

        # 4. Sudden deceleration on either vehicle
        for t in (t1, t2):
            drop = t.speed_drop_ratio
            if drop >= cfg.speed_drop_threshold:
                score += 0.15
                signals.append(
                    f"speed_drop={drop:.0%} on track #{t.track_id}"
                )

        # 5. Heading divergence (objects bouncing apart after collision)
        if t1.speed > 0.5 and t2.speed > 0.5:
            hdiff = _heading_diff(t1.heading, t2.heading)
            if hdiff > cfg.direction_change_threshold:
                contribution = min(1.0, (hdiff - cfg.direction_change_threshold) / 90.0)
                score += 0.15 * contribution
                signals.append(f"heading_divergence={hdiff:.0f}deg")

        # 6. Person near collision zone
        if persons:
            cx_pair = (t1.cx + t2.cx) / 2
            cy_pair = (t1.cy + t2.cy) / 2
            for p in persons:
                pd = math.hypot(p.cx - cx_pair, p.cy - cy_pair)
                if pd < cfg.proximity_threshold_px * 1.5:
                    score += 0.10
                    etype = EventType.VEHICLE_PEDESTRIAN
                    signals.append(
                        f"person #{p.track_id} near collision zone "
                        f"(distance={pd:.0f}px)"
                    )
                    break

        return min(1.0, score), signals, etype

    def _score_sudden_stop(
        self, t: Track
    ) -> Tuple[float, List[str]]:
        """
        Scores a single vehicle for a sudden-stop event.
        This may indicate collision with a stationary object.
        """
        score = 0.0
        signals: List[str] = []

        if len(t.speed_history) < 5:
            return 0.0, []

        drop = t.speed_drop_ratio
        if drop >= cfg.speed_drop_threshold * 1.2:
            score += 0.40
            signals.append(f"sudden_stop: speed_drop={drop:.0%} on track #{t.track_id}")

        # Was moving fast before the stop?
        max_recent = max(t.speed_history[:-2], default=0.0)
        if max_recent > 5.0 and t.speed < 1.0:
            score += 0.30
            signals.append(
                f"high-speed_to_stationary: prev_max={max_recent:.1f}px/frame "
                f"now={t.speed:.1f}px/frame on track #{t.track_id}"
            )

        return min(1.0, score), signals

    def reset(self) -> None:
        """Resets the detector state (call when starting a new video)."""
        self._frame_buffer.clear()
        self._consecutive_detections = 0
        self._last_event_frame = -999
        self._frame_index = 0

    @property
    def consecutive_count(self) -> int:
        """Number of consecutive frames where threshold was exceeded."""
        return self._consecutive_detections
