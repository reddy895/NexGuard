"""
NexGuard Object Tracking & Motion Estimation Subsystem
"""

import math
from typing import List, Dict, Tuple, Optional
import numpy as np

from backend.app.detection.detector import DetectionResult


class TrackedObject:
    def __init__(self, track_id: int, detection: DetectionResult):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.center = detection.center
        
        # History of centroid positions: [(x, y), ...]
        self.history: List[Tuple[float, float]] = [tuple(self.center)]
        self.max_history = 30
        
        # Motion metrics
        self.velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/frame
        self.speed: float = 0.0                          # speed magnitude
        self.direction_deg: float = 0.0                  # motion heading angle in degrees
        self.frames_tracked: int = 1
        self.missing_frames: int = 0

    def update(self, detection: DetectionResult):
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        prev_center = self.center
        self.center = detection.center
        
        # Calculate instant velocity vector
        vx = self.center[0] - prev_center[0]
        vy = self.center[1] - prev_center[1]
        
        # Exponential moving average smoothing for velocity
        alpha = 0.6
        self.velocity = (
            alpha * vx + (1 - alpha) * self.velocity[0],
            alpha * vy + (1 - alpha) * self.velocity[1]
        )
        self.speed = math.hypot(self.velocity[0], self.velocity[1])
        if self.speed > 0.1:
            self.direction_deg = math.degrees(math.atan2(self.velocity[1], self.velocity[0]))
            
        self.history.append(tuple(self.center))
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        self.frames_tracked += 1
        self.missing_frames = 0
        
        # Attach track ID back to detection
        detection.track_id = self.track_id

    def predict_next_position(self) -> Tuple[float, float]:
        """Predicts position in next frame using velocity vector."""
        return (
            self.center[0] + self.velocity[0],
            self.center[1] + self.velocity[1]
        )

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(x, 2) for x in self.bbox],
            "center": [round(x, 2) for x in self.center],
            "speed_px": round(self.speed, 2),
            "velocity": [round(v, 2) for v in self.velocity],
            "frames_tracked": self.frames_tracked
        }


class NexGuardTracker:
    def __init__(self, max_distance_px: float = 80.0, max_missing_frames: int = 10):
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_distance_px = max_distance_px
        self.max_missing_frames = max_missing_frames

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: List[DetectionResult]) -> List[TrackedObject]:
        """Matches current detections to existing tracks using distance and class consistency."""
        if not self.tracks:
            # First frame or reset: register all detections as new tracks
            for det in detections:
                track = TrackedObject(self.next_id, det)
                det.track_id = self.next_id
                self.tracks[self.next_id] = track
                self.next_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        active_tracks = [self.tracks[tid] for tid in track_ids]

        # Calculate cost matrix (distance between predicted track position and detection center)
        num_tracks = len(active_tracks)
        num_dets = len(detections)

        if num_dets == 0:
            # Mark all as missing
            for t in active_tracks:
                t.missing_frames += 1
            self._purge_stale_tracks()
            return [t for t in self.tracks.values() if t.missing_frames == 0]

        cost_matrix = np.full((num_tracks, num_dets), fill_value=1e6)

        for i, track in enumerate(active_tracks):
            pred_x, pred_y = track.predict_next_position()
            for j, det in enumerate(detections):
                # Must match class family (e.g. vehicle to vehicle, person to person)
                if track.class_name != det.class_name:
                    continue
                dist = math.hypot(pred_x - det.center[0], pred_y - det.center[1])
                if dist <= self.max_distance_px:
                    cost_matrix[i, j] = dist

        assigned_tracks = set()
        assigned_dets = set()

        # Greedy distance matching
        while True:
            min_val = cost_matrix.min()
            if min_val >= self.max_distance_px or min_val == 1e6:
                break
            i, j = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
            
            track = active_tracks[i]
            det = detections[j]
            track.update(det)
            
            assigned_tracks.add(track.track_id)
            assigned_dets.add(j)
            
            cost_matrix[i, :] = 1e6
            cost_matrix[:, j] = 1e6

        # Mark unassigned tracks as missing
        for i, track in enumerate(active_tracks):
            if track.track_id not in assigned_tracks:
                track.missing_frames += 1

        # Register unassigned detections as new tracks
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                track = TrackedObject(self.next_id, det)
                det.track_id = self.next_id
                self.tracks[self.next_id] = track
                self.next_id += 1

        self._purge_stale_tracks()
        return [t for t in self.tracks.values() if t.missing_frames == 0]

    def _purge_stale_tracks(self):
        stale = [tid for tid, t in self.tracks.items() if t.missing_frames > self.max_missing_frames]
        for tid in stale:
            del self.tracks[tid]
