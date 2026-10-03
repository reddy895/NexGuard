"""
NexGuard Persistent Object Tracking & Track History Module
Tracks vehicles and pedestrians across consecutive frames, calculating velocity,
acceleration, direction, track age, and center history buffer.
"""

import time
import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from config import config
from src.detector import DetectionObject
from utils.geometry import (
    get_bbox_center,
    compute_iou,
    compute_euclidean_distance,
    compute_vector_magnitude,
    compute_angle_degrees
)


class TrackedObject:
    """Represents an object actively tracked across video frames."""

    def __init__(self, track_id: int, detection: DetectionObject, max_history: int = 20):
        self.track_id: int = track_id
        self.class_name: str = detection.class_name
        self.confidence: float = detection.confidence
        self.bbox: List[float] = detection.bbox
        self.center: Tuple[float, float] = detection.center
        
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.speed: float = 0.0
        self.acceleration: Tuple[float, float] = (0.0, 0.0)
        self.accel_magnitude: float = 0.0
        self.direction_angle: float = 0.0

        self.frames_tracked: int = 1
        self.lost_frames: int = 0
        self.last_seen: float = time.time()
        self.max_history: int = max_history

        self.center_history: List[Tuple[float, float]] = [self.center]
        self.bbox_history: List[List[float]] = [self.bbox]

    def update(self, detection: DetectionObject):
        """Updates track position, calculates velocity, acceleration, and appends to history."""
        old_center = self.center
        old_velocity = self.velocity

        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.center = detection.center
        self.last_seen = time.time()
        self.lost_frames = 0
        self.frames_tracked += 1

        # Append to history
        self.center_history.append(self.center)
        self.bbox_history.append(self.bbox)
        if len(self.center_history) > self.max_history:
            self.center_history.pop(0)
            self.bbox_history.pop(0)

        # Estimate Velocity (px/frame)
        vx = self.center[0] - old_center[0]
        vy = self.center[1] - old_center[1]
        self.velocity = (vx, vy)
        self.speed = compute_vector_magnitude(self.velocity)
        self.direction_angle = compute_angle_degrees(self.velocity)

        # Estimate Acceleration (px/frame²)
        ax = vx - old_velocity[0]
        ay = vy - old_velocity[1]
        self.acceleration = (ax, ay)
        self.accel_magnitude = compute_vector_magnitude(self.acceleration)

    def mark_lost(self):
        """Increments lost frames count when track is unobserved in current frame."""
        self.lost_frames += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(b, 2) for b in self.bbox],
            "center": (round(self.center[0], 2), round(self.center[1], 2)),
            "velocity": (round(self.velocity[0], 2), round(self.velocity[1], 2)),
            "speed": round(self.speed, 2),
            "acceleration": (round(self.acceleration[0], 2), round(self.acceleration[1], 2)),
            "frames_tracked": self.frames_tracked,
            "direction_angle": round(self.direction_angle, 1)
        }


class ObjectTracker:
    """Multi-object persistent tracker based on spatial IoU and Euclidean distance matching."""

    def __init__(self, max_lost_frames: int = 30, iou_threshold: float = 0.25):
        self.next_track_id: int = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_lost_frames: int = max_lost_frames
        self.iou_threshold: float = iou_threshold
        self.history_length: int = config.track_history_length

    def reset(self):
        """Resets tracker state."""
        self.next_track_id = 1
        self.tracks.clear()

    def update(self, detections: List[DetectionObject]) -> List[TrackedObject]:
        """Matches incoming detections to active tracks, creates new tracks, and prunes lost tracks."""
        active_track_ids = list(self.tracks.keys())
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = set(active_track_ids)

        if active_track_ids and detections:
            # Build cost matrix based on IoU and Center Distance
            cost_matrix = np.zeros((len(active_track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(active_track_ids):
                track = self.tracks[tid]
                for j, det in enumerate(detections):
                    if track.class_name == det.class_name or (
                        track.class_name in config.vehicle_classes and det.class_name in config.vehicle_classes
                    ):
                        iou = compute_iou(track.bbox, det.bbox)
                        dist = compute_euclidean_distance(track.center, det.center)
                        # Combine IoU and distance score (0.0 to 1.0)
                        dist_score = max(0.0, 1.0 - (dist / config.proximity_threshold_px))
                        score = (iou * 0.7) + (dist_score * 0.3)
                        cost_matrix[i, j] = score
                    else:
                        cost_matrix[i, j] = 0.0

            # Match greedily based on highest matching score
            matched_pairs = []
            while True:
                if cost_matrix.size == 0 or np.max(cost_matrix) < 0.15:
                    break
                max_idx = np.unravel_index(np.argmax(cost_matrix, axis=None), cost_matrix.shape)
                i, j = max_idx
                if cost_matrix[i, j] < 0.15:
                    break

                tid = active_track_ids[i]
                matched_pairs.append((tid, j))
                cost_matrix[i, :] = -1.0
                cost_matrix[:, j] = -1.0

            # Update matched tracks
            for tid, det_idx in matched_pairs:
                self.tracks[tid].update(detections[det_idx])
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if tid in unmatched_tracks:
                    unmatched_tracks.remove(tid)

        # Mark unmatched tracks as lost
        for tid in unmatched_tracks:
            self.tracks[tid].mark_lost()

        # Prune tracks lost for too long
        dead_tracks = [tid for tid, t in self.tracks.items() if t.lost_frames > self.max_lost_frames]
        for tid in dead_tracks:
            del self.tracks[tid]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = TrackedObject(
                track_id=self.next_track_id,
                detection=det,
                max_history=self.history_length
            )
            self.tracks[self.next_track_id] = new_track
            self.next_track_id += 1

        return [t for t in self.tracks.values() if t.lost_frames == 0]
