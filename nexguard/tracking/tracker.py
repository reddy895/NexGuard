"""
NexGuard — Object Tracking Engine
===================================
Multi-object tracker using IoU-based greedy association.

Architecture:
    ObjectTracker (public interface)
        └── _IoUTracker (internal implementation)

The _IoUTracker uses frame-to-frame IoU matching as its primary
association strategy, augmented by centroid distance for low-overlap
scenarios. Works reliably for fixed-camera surveillance footage
without requiring heavy external dependencies.

Future extension:
    To integrate ByteTrack or BOT-SORT, replace _IoUTracker.__init__
    in ObjectTracker and convert the results to Track objects.
    The accident detector and overlay layer remain unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import cfg
from nexguard.models.schemas import Detection, Track
from nexguard.utils.logging import get_logger
from nexguard.utils.utils import bbox_iou, bbox_centroid_dist

log = get_logger("nexguard.tracker")


# ---------------------------------------------------------------------------
# Internal IoU tracker
# ---------------------------------------------------------------------------

class _IoUTracker:
    """
    Greedy IoU-based multi-object tracker.

    Associates detections to existing tracks using IoU as the primary
    metric and centroid distance as the fallback. Tracks persist for up
    to `max_lost_frames` without a detection match before removal.
    """

    def __init__(
        self,
        max_lost_frames: int = 30,
        iou_threshold: float = 0.20,
        max_dist_px: float = 150.0,
        history_length: int = 30,
    ) -> None:
        self._max_lost = max_lost_frames
        self._iou_thresh = iou_threshold
        self._max_dist = max_dist_px
        self._history_len = history_length
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Associates new detections to existing tracks.

        Args:
            detections: Detection objects for the current frame.

        Returns:
            All currently active Track objects.
        """
        # Mark all tracks as one frame older / potentially lost
        for t in self._tracks.values():
            t.lost_frames += 1
            t.age += 1
            # Linearly extrapolate position for short gaps
            if t.lost_frames > 1 and t.speed > 0.5:
                vx, vy = t.velocity
                damping = 0.8 ** min(t.lost_frames, 5)
                dx, dy = vx * damping, vy * damping
                t.cx += dx
                t.cy += dy
                x1, y1, x2, y2 = t.bbox
                t.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

        matched_track_ids: set = set()
        matched_det_indices: set = set()

        if self._tracks and detections:
            track_list = list(self._tracks.values())
            n_t, n_d = len(track_list), len(detections)

            # Build cost matrix: IoU range [0, 0.80], centroid range [0.80, 0.99]
            cost = np.full((n_t, n_d), 2.0, dtype=np.float64)
            for i, track in enumerate(track_list):
                for j, det in enumerate(detections):
                    iou_val = bbox_iou(track.bbox, det.bbox)
                    dist = bbox_centroid_dist(track.bbox, det.bbox)
                    if iou_val >= self._iou_thresh:
                        cost[i, j] = (1.0 - iou_val) * 0.8
                    elif dist <= self._max_dist:
                        cost[i, j] = 0.80 + 0.19 * min(dist / max(self._max_dist, 1.0), 1.0)

            # Greedy assignment: lowest cost first
            for flat_idx in np.argsort(cost.ravel()):
                i, j = divmod(int(flat_idx), n_d)
                if i >= n_t or j >= n_d:
                    continue
                if cost[i, j] >= 1.0:
                    break
                track = track_list[i]
                if track.track_id in matched_track_ids:
                    continue
                if j in matched_det_indices:
                    continue
                det = detections[j]
                # Class compatibility: same class, or both in the vehicle category
                if not (
                    track.class_name == det.class_name
                    or (track.is_vehicle and det.class_name in cfg.vehicle_classes)
                ):
                    continue
                self._update_track(track, det)
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(j)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                self._create_track(det)

        # Prune stale tracks
        for tid in [tid for tid, t in self._tracks.items() if t.lost_frames > self._max_lost]:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _create_track(self, det: Detection) -> Track:
        cx = (det.bbox[0] + det.bbox[2]) / 2.0
        cy = (det.bbox[1] + det.bbox[3]) / 2.0
        track = Track(
            track_id=self._next_id,
            bbox=det.bbox,
            class_name=det.class_name,
            confidence=det.confidence,
            cx=cx,
            cy=cy,
            history=[(cx, cy)],
            speed_history=[0.0],
            lost_frames=0,
            age=1,
            hits=1,
            is_vehicle=det.class_name in cfg.vehicle_classes,
            is_person=det.class_name in cfg.person_classes,
            is_accident=det.is_accident or (det.class_name in cfg.accident_classes),
        )
        self._tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _update_track(self, track: Track, det: Detection) -> None:
        new_cx = (det.bbox[0] + det.bbox[2]) / 2.0
        new_cy = (det.bbox[1] + det.bbox[3]) / 2.0
        vx = new_cx - track.cx
        vy = new_cy - track.cy
        speed = math.hypot(vx, vy)

        track.bbox = det.bbox
        track.class_name = det.class_name
        track.is_vehicle = det.class_name in cfg.vehicle_classes
        track.is_person = det.class_name in cfg.person_classes
        track.is_accident = det.is_accident or (det.class_name in cfg.accident_classes)
        track.confidence = det.confidence
        if track.hits > 1 and track.velocity != (0.0, 0.0):
            vx = 0.70 * vx + 0.30 * track.velocity[0]
            vy = 0.70 * vy + 0.30 * track.velocity[1]
            speed = math.hypot(vx, vy)
        track.velocity = (vx, vy)
        track.speed = speed
        track.heading = math.degrees(math.atan2(vy, vx))
        track.cx = new_cx
        track.cy = new_cy
        track.lost_frames = 0
        track.hits += 1

        track.speed_history.append(speed)
        if len(track.speed_history) > self._history_len:
            track.speed_history.pop(0)

        track.history.append((new_cx, new_cy))
        if len(track.history) > self._history_len:
            track.history.pop(0)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    Public tracking interface used by the NexGuard pipeline.

    Wraps _IoUTracker and provides the API expected by the
    AccidentDetector and overlay layer. Swap the internal
    implementation here to upgrade to ByteTrack etc.
    """

    def __init__(
        self,
        max_lost_frames: Optional[int] = None,
        iou_threshold: Optional[float] = None,
        history_length: Optional[int] = None,
    ) -> None:
        self._impl = _IoUTracker(
            max_lost_frames=max_lost_frames or cfg.max_lost_frames,
            iou_threshold=iou_threshold or 0.20,
            history_length=history_length or cfg.track_history_length,
        )
        log.info(
            "Tracker ready — max_lost=%d history=%d",
            cfg.max_lost_frames, cfg.track_history_length,
        )

    def update(self, detections: List[Detection]) -> List[Track]:
        """Processes detections and returns the current set of active tracks."""
        return self._impl.update(detections)

    def get_tracks(self) -> List[Track]:
        return list(self._impl._tracks.values())

    def get_vehicles(self) -> List[Track]:
        return [t for t in self._impl._tracks.values() if t.is_vehicle]

    def get_persons(self) -> List[Track]:
        return [t for t in self._impl._tracks.values() if t.is_person]

    def reset(self) -> None:
        """Resets all tracking state (call when starting a new source)."""
        self._impl.reset()
        log.info("Tracker reset.")

    @property
    def active_count(self) -> int:
        return len(self._impl._tracks)
