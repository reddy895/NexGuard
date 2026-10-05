"""
NexGuard — Object Tracking Engine
===================================
Provides a clean abstraction layer for multi-object tracking.

Architecture:
    ObjectTracker (public interface)
        └── IoUTracker (default implementation — no external deps required)

The IoUTracker uses frame-to-frame IoU matching as its primary association
strategy, augmented by centroid distance for low-overlap scenarios.
This approach works reliably for surveillance video with a fixed camera
and avoids complex dependencies while still supporting the accident
detection pipeline's temporal requirements.

Future extension:
    To integrate ByteTrack or BOT-SORT, override ObjectTracker.update()
    to call the Ultralytics built-in tracker and convert results into
    Track objects. The rest of the pipeline remains unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import cfg
from nexguard.models.schemas import Detection, Track
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.tracking")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Intersection over Union for two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _centroid_dist(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Euclidean distance between bounding-box centers."""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.hypot(cx2 - cx1, cy2 - cy1)


# ---------------------------------------------------------------------------
# IoU-based tracker implementation
# ---------------------------------------------------------------------------

class _IoUTracker:
    """
    Greedy IoU-based multi-object tracker.

    Associates detections to existing tracks using IoU as the primary
    similarity metric. Tracks persist for up to `max_lost_frames` frames
    without a detection before being removed.
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
            detections: Detections from the current frame.

        Returns:
            List of currently active Track objects.
        """
        # -- Mark all tracks as potentially lost this frame
        for t in self._tracks.values():
            t.lost_frames += 1

        matched_track_ids = set()
        matched_det_indices = set()

        if self._tracks and detections:
            track_list = list(self._tracks.values())
            n_tracks = len(track_list)
            n_dets = len(detections)

            # Build cost matrix (1 - IoU) with distance penalty
            cost = np.ones((n_tracks, n_dets), dtype=np.float64)
            for i, track in enumerate(track_list):
                for j, det in enumerate(detections):
                    iou_val = _iou(track.bbox, det.bbox)
                    dist = _centroid_dist(track.bbox, det.bbox)
                    if iou_val >= self._iou_thresh or dist <= self._max_dist:
                        cost[i, j] = 1.0 - iou_val

            # Greedy matching — best IoU first
            flat_order = np.argsort(cost.ravel())
            for flat_idx in flat_order:
                i = flat_idx // n_dets
                j = flat_idx % n_dets
                if i >= n_tracks or j >= len(detections):
                    break
                if cost[i, j] >= 1.0:
                    break
                track = track_list[i]
                if track.track_id in matched_track_ids:
                    continue
                if j in matched_det_indices:
                    continue
                det = detections[j]
                if track.class_name != det.class_name:
                    continue
                self._update_track(track, det)
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(j)

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_det_indices:
                self._create_track(det)

        # Remove stale tracks
        stale = [
            tid
            for tid, t in self._tracks.items()
            if t.lost_frames > self._max_lost
        ]
        for tid in stale:
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
            is_vehicle=det.class_name in cfg.vehicle_classes,
            is_person=det.class_name in cfg.person_classes,
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
        heading = math.degrees(math.atan2(vy, vx))

        track.bbox = det.bbox
        track.confidence = det.confidence
        track.velocity = (vx, vy)
        track.speed = speed
        track.heading = heading
        track.cx = new_cx
        track.cy = new_cy
        track.lost_frames = 0
        track.age += 1

        track.speed_history.append(speed)
        if len(track.speed_history) > self._history_len:
            track.speed_history.pop(0)

        track.history.append((new_cx, new_cy))
        if len(track.history) > self._history_len:
            track.history.pop(0)

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def get_track(self, track_id: int) -> Optional[Track]:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ---------------------------------------------------------------------------
# Public tracker interface
# ---------------------------------------------------------------------------

class ObjectTracker:
    """
    Public tracking interface used by the NexGuard pipeline.

    Wraps the internal _IoUTracker and provides the API expected
    by the accident detector and visualization layer.

    To swap in a different tracking backend (ByteTrack, etc.),
    replace the _impl initialization in __init__ and keep this
    interface unchanged.
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
            f"Tracker initialized — max_lost={cfg.max_lost_frames} "
            f"history={cfg.track_history_length}"
        )

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Processes new detections and returns the current set of active tracks.

        Args:
            detections: Detection objects from the current frame.

        Returns:
            All currently active Track objects.
        """
        return self._impl.update(detections)

    def get_tracks(self) -> List[Track]:
        """Returns all currently active tracks."""
        return list(self._impl.tracks.values())

    def get_track(self, track_id: int) -> Optional[Track]:
        """Returns a specific track by ID, or None if not found."""
        return self._impl.get_track(track_id)

    def get_vehicles(self) -> List[Track]:
        """Returns only vehicle tracks."""
        return [t for t in self._impl.tracks.values() if t.is_vehicle]

    def get_persons(self) -> List[Track]:
        """Returns only person tracks."""
        return [t for t in self._impl.tracks.values() if t.is_person]

    def remove_stale_tracks(self) -> None:
        """Manually triggers removal of lost tracks (called automatically on update)."""
        stale = [
            tid
            for tid, t in self._impl.tracks.items()
            if t.lost_frames > self._impl._max_lost
        ]
        for tid in stale:
            del self._impl._tracks[tid]

    def reset(self) -> None:
        """Resets the tracker, clearing all tracks."""
        self._impl.reset()
        log.info("Tracker reset.")

    @property
    def active_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._impl.tracks)
