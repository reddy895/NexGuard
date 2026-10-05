"""
NexGuard Multi-Object Tracking Engine
Tracks detected vehicles and pedestrians, computes velocity vectors, sudden deceleration,
and spatial proximity metrics.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple


class TrackedObject:
    """Represents a single tracked entity (vehicle or person)."""
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], class_name: str, confidence: float):
        self.track_id = track_id
        self.bbox = bbox  # (xmin, ymin, xmax, ymax)
        self.class_name = class_name
        self.confidence = confidence
        
        # Center coordinate calculation
        self.cx = (bbox[0] + bbox[2]) / 2.0
        self.cy = (bbox[1] + bbox[3]) / 2.0
        
        self.history: List[Tuple[float, float]] = [(self.cx, self.cy)]
        self.lost_frames: int = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) px/frame
        self.speed: float = 0.0                           # magnitude px/frame
        self.heading_angle: float = 0.0                    # degrees
        self.speed_history: List[float] = [0.0]

    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Updates object location and recalculates velocity and speed drop dynamics."""
        self.bbox = bbox
        self.confidence = confidence
        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0
        
        vx = new_cx - self.cx
        vy = new_cy - self.cy
        self.velocity = (vx, vy)
        
        self.speed = float(np.hypot(vx, vy))
        self.speed_history.append(self.speed)
        if len(self.speed_history) > 10:
            self.speed_history.pop(0)

        self.heading_angle = math.degrees(math.atan2(vy, vx))
        
        self.cx = new_cx
        self.cy = new_cy
        self.history.append((self.cx, self.cy))
        if len(self.history) > 20:
            self.history.pop(0)
            
    def get_acceleration(self) -> float:
        """Computes rate of speed change (px/frame^2)."""
        if len(self.speed_history) < 2:
            return 0.0
        return self.speed_history[-1] - self.speed_history[-2]

    def get_speed_drop_ratio(self) -> float:
        """Computes relative speed drop ratio between recent peak and current speed."""
        if len(self.speed_history) < 3:
            return 0.0
        peak_speed = max(self.speed_history[:-1])
        if peak_speed <= 0.5:
            return 0.0
        current_speed = self.speed_history[-1]
        drop = max(0.0, peak_speed - current_speed)
        return drop / peak_speed


class ObjectTracker:
    """IoU and Distance-based Multi-Object Tracker."""
    def __init__(self, max_lost_frames: int = 25, iou_threshold: float = 0.3):
        self.next_id = 1
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = boxAArea + boxBArea - interArea
        if denominator <= 0:
            return 0.0
        return interArea / denominator

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """Updates tracks with new detections from current frame."""
        # Increment lost frames for existing tracks
        for obj in self.tracked_objects.values():
            obj.lost_frames += 1

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracked_objects.keys())

        # Match using IoU matrix
        if unmatched_tracks and unmatched_detections:
            iou_matrix = np.zeros((len(unmatched_tracks), len(unmatched_detections)))
            for i, tid in enumerate(unmatched_tracks):
                for j, det_idx in enumerate(unmatched_detections):
                    det_box = detections[det_idx]["bbox"]
                    iou_matrix[i, j] = self.compute_iou(self.tracked_objects[tid].bbox, det_box)

            # Greedy matching
            matched_pairs = []
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = unmatched_tracks[i]
                det_idx = unmatched_detections[j]
                
                self.tracked_objects[tid].update(
                    bbox=detections[det_idx]["bbox"],
                    confidence=detections[det_idx]["confidence"]
                )
                
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

                if tid in unmatched_tracks:
                    unmatched_tracks.remove(tid)
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)

        # Create new tracks for remaining unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_obj = TrackedObject(
                track_id=self.next_id,
                bbox=det["bbox"],
                class_name=det["class_name"],
                confidence=det["confidence"]
            )
            self.tracked_objects[self.next_id] = new_obj
            self.next_id += 1

        # Remove dead tracks exceeding max_lost_frames
        dead_ids = [tid for tid, obj in self.tracked_objects.items() if obj.lost_frames > self.max_lost_frames]
        for tid in dead_ids:
            del self.tracked_objects[tid]

        return list(self.tracked_objects.values())

    def reset(self):
        """Clears all tracking state."""
        self.next_id = 1
        self.tracked_objects.clear()
