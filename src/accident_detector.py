"""Temporal accident analysis and evidence persistence for NexGuard."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import cv2
import numpy as np

from src.severity_analyzer import SeverityAnalyzer


def build_detection_record(
    track_id: Optional[int],
    class_id: int,
    class_name: str,
    confidence: float,
    box: Sequence[float],
    center: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Normalize a raw detection into a tracked object record."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if center is None:
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    record: Dict[str, Any] = {
        "track_id": track_id,
        "class_id": int(class_id),
        "class_name": str(class_name),
        "confidence": round(float(confidence), 4),
        "bbox": [x1, y1, x2, y2],
        "center": [float(center[0]), float(center[1])],
        "velocity": [0.0, 0.0],
        "history": [[float(center[0]), float(center[1])]],
    }
    return record


class AccidentDetector:
    """Build a temporal accident-analysis layer over tracked YOLO detections.

    The base YOLOv8 COCO model detects people and vehicles, but it does not directly
    classify "accident" events. NexGuard therefore combines object detection,
    tracking, and multiple-frame motion/spatial signals to infer likely collisions.
    """

    PERSON_CLASSES = ("person",)
    VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")

    def __init__(
        self,
        candidate_threshold: float = 0.55,
        confirmation_frames: int = 5,
        cooldown_frames: int = 100,
        max_history: int = 20,
        output_dir: str = "incidents",
    ) -> None:
        self.candidate_threshold = float(candidate_threshold)
        self.confirmation_frames = int(max(1, confirmation_frames))
        self.cooldown_frames = int(max(0, cooldown_frames))
        self.max_history = int(max(1, max_history))
        self.output_dir = output_dir

        self.tracked_objects: Dict[int, Dict[str, Any]] = {}
        self.track_history: Dict[int, List[List[float]]] = {}
        self.next_track_id = 1
        self.candidate_frames = 0
        self.cooldown_remaining = 0
        self.last_candidate_score = 0.0
        self.last_event: Optional[Dict[str, Any]] = None

    def _next_id(self) -> int:
        next_id = self.next_track_id
        self.next_track_id += 1
        return next_id

    def _normalize_object(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        class_name = str(detection.get("class_name", "unknown")).lower()
        box = detection.get("bbox") or detection.get("box") or [0.0, 0.0, 0.0, 0.0]
        confidence = float(detection.get("confidence", 0.0))
        center = detection.get("center")
        if center is None:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        track_id = detection.get("track_id")
        if track_id is None:
            track_id = self._next_id()

        obj = build_detection_record(
            track_id=track_id,
            class_id=int(detection.get("class_id", 0)),
            class_name=class_name,
            confidence=confidence,
            box=box,
            center=center,
        )
        obj["velocity"] = detection.get("velocity", [0.0, 0.0])
        return obj

    def update_tracks(self, detections: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update track state with the latest frame results."""
        current_objects: List[Dict[str, Any]] = []

        for detection in detections:
            obj = self._normalize_object(detection)
            track_id = int(obj["track_id"])
            prev_obj = self.tracked_objects.get(track_id)

            if prev_obj is not None:
                prev_center = prev_obj.get("center", obj["center"])
                dx = obj["center"][0] - prev_center[0]
                dy = obj["center"][1] - prev_center[1]
                obj["velocity"] = [float(dx), float(dy)]
            else:
                obj["velocity"] = [0.0, 0.0]

            history = self.track_history.setdefault(track_id, [])
            history.append(obj["center"])
            if len(history) > self.max_history:
                history.pop(0)
            obj["history"] = history[:]

            self.tracked_objects[track_id] = obj
            current_objects.append(obj)

        for track_id in list(self.tracked_objects.keys()):
            if track_id not in {int(item["track_id"]) for item in current_objects}:
                self.tracked_objects.pop(track_id, None)
                self.track_history.pop(track_id, None)

        return sorted(current_objects, key=lambda item: int(item["track_id"]))

    @staticmethod
    def _box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
        x1_a, y1_a, x2_a, y2_a = [float(v) for v in box_a[:4]]
        x1_b, y1_b, x2_b, y2_b = [float(v) for v in box_b[:4]]

        x1 = max(x1_a, x1_b)
        y1 = max(y1_a, y1_b)
        x2 = min(x2_a, x2_b)
        y2 = min(y2_a, y2_b)

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, (x2_a - x1_a) * (y2_a - y1_a))
        area_b = max(0.0, (x2_b - x1_b) * (y2_b - y1_b))
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    @staticmethod
    def _distance_between(center_a: Sequence[float], center_b: Sequence[float]) -> float:
        dx = float(center_b[0]) - float(center_a[0])
        dy = float(center_b[1]) - float(center_a[1])
        return math.hypot(dx, dy)

    def _evaluate_pair(self, obj_a: Dict[str, Any], obj_b: Dict[str, Any], frame_shape: Sequence[int]) -> Tuple[float, int, List[str]]:
        if obj_a.get("class_name", "unknown") not in self.VEHICLE_CLASSES or obj_b.get("class_name", "unknown") not in self.VEHICLE_CLASSES:
            return 0.0, 0, []

        height, width = frame_shape[:2]
        frame_diag = math.hypot(width, height)
        center_a = obj_a.get("center", [0.0, 0.0])
        center_b = obj_b.get("center", [0.0, 0.0])

        distance = self._distance_between(center_a, center_b)
        proximity_score = max(0.0, 1.0 - (distance / max(frame_diag * 0.5, 1.0)))

        vel_a = obj_a.get("velocity", [0.0, 0.0])
        vel_b = obj_b.get("velocity", [0.0, 0.0])
        relative_velocity = [vel_a[0] - vel_b[0], vel_a[1] - vel_b[1]]
        closing_speed = math.hypot(*relative_velocity)
        closing_score = min(1.0, closing_speed / 40.0)

        history_a = obj_a.get("history", [])
        history_b = obj_b.get("history", [])
        if len(history_a) >= 2 and len(history_b) >= 2:
            prev_a = history_a[-2]
            prev_b = history_b[-2]
            prev_a_vel = [center_a[0] - prev_a[0], center_a[1] - prev_a[1]]
            prev_b_vel = [center_b[0] - prev_b[0], center_b[1] - prev_b[1]]
            motion_change = abs((math.hypot(*vel_a) - math.hypot(*prev_a_vel))) + abs((math.hypot(*vel_b) - math.hypot(*prev_b_vel)))
        else:
            motion_change = math.hypot(*vel_a) + math.hypot(*vel_b)
        motion_score = min(1.0, motion_change / 80.0)

        overlap_score = self._box_iou(obj_a.get("bbox", [0, 0, 0, 0]), obj_b.get("bbox", [0, 0, 0, 0]))
        person_signal = 0.0
        nearby_people: List[str] = []
        for person in self.tracked_objects.values():
            if person.get("class_name", "unknown") not in self.PERSON_CLASSES:
                continue
            person_center = person.get("center", [0.0, 0.0])
            if self._distance_between(center_a, person_center) < 100 or self._distance_between(center_b, person_center) < 100:
                nearby_people.append(person.get("class_name", "person"))
                person_signal = 1.0
        signals = [
            (proximity_score >= 0.35, "proximity"),
            (closing_score >= 0.25, "closing_speed"),
            (overlap_score >= 0.05 or distance < 50, "overlap"),
            (motion_score >= 0.35, "motion_change"),
            (person_signal >= 0.5, "person_involvement"),
        ]
        active_signals = [label for triggered, label in signals if triggered]
        score = (
            proximity_score
            + closing_score
            + max(overlap_score, 0.0)
            + motion_score
            + person_signal
        ) / 5.0
        return score, len(active_signals), active_signals

    def analyze_frame(
        self,
        tracked_objects: Sequence[Dict[str, Any]],
        frame_shape: Sequence[int] = (480, 640),
        frame: Optional[np.ndarray] = None,
        video_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate the track set for a potential accident and return an event dict when confirmed."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
            if self.last_event is not None:
                return None

        if not tracked_objects:
            return None

        objects = list(tracked_objects)
        vehicles = [obj for obj in objects if obj.get("class_name", "unknown").lower() in self.VEHICLE_CLASSES]
        if len(vehicles) < 2:
            self.candidate_frames = 0
            self.last_candidate_score = 0.0
            return None

        best_score = 0.0
        best_pair: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        best_signals: List[str] = []

        for idx, first in enumerate(vehicles):
            for second in vehicles[idx + 1 :]:
                score, signal_count, signals = self._evaluate_pair(first, second, frame_shape)
                if score > best_score:
                    best_score = score
                    best_pair = (first, second)
                    best_signals = signals

        has_strong_signals = len(best_signals) >= 3
        if best_pair is None or (best_score < self.candidate_threshold and not has_strong_signals):
            self.candidate_frames = 0
            self.last_candidate_score = 0.0
            return None

        self.last_candidate_score = best_score
        self.candidate_frames += 1

        if self.candidate_frames < self.confirmation_frames:
            return {
                "status": "suspected",
                "confidence": round(float(best_score), 4),
                "candidate_score": float(best_score),
                "signals": best_signals,
            }

        if self.cooldown_remaining > 0:
            return None

        first, second = best_pair
        vehicles_involved = 2
        people_nearby = 0
        for obj in objects:
            if obj.get("class_name", "unknown").lower() in self.PERSON_CLASSES:
                person_center = obj.get("center", [0.0, 0.0])
                if self._distance_between(first.get("center", [0.0, 0.0]), person_center) < 120 or self._distance_between(second.get("center", [0.0, 0.0]), person_center) < 120:
                    people_nearby += 1

        severity = SeverityAnalyzer.compute(best_score, vehicles_involved, people_nearby)
        event = {
            "event_type": "accident",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "confidence": round(float(best_score), 4),
            "vehicles_involved": vehicles_involved,
            "persons_nearby": people_nearby,
            "location": None,
            "frame_path": None,
            "video_path": video_path,
            "status": "confirmed",
            "signals": best_signals,
            "evidence_path": None,
        }

        if frame is None and frame_shape is not None:
            height, width = frame_shape[:2]
            frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)

        if frame is not None:
            event["evidence_path"] = self.save_incident_evidence(frame, event)
            event["frame_path"] = event["evidence_path"]

        self.last_event = event
        self.cooldown_remaining = self.cooldown_frames
        self.candidate_frames = 0
        return event

    def save_incident_evidence(self, frame: np.ndarray, event: Dict[str, Any]) -> str:
        """Persist the incident frame and metadata to disk without overwriting prior evidence."""
        if frame is None:
            return ""

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = uuid4().hex[:8]
        incident_dir = Path(self.output_dir) / f"incident_{timestamp}_{unique_id}"
        incident_dir.mkdir(parents=True, exist_ok=True)

        image_path = incident_dir / "accident_frame.jpg"
        metadata_path = incident_dir / "incident_metadata.json"

        cv2.imwrite(str(image_path), frame)
        visible_event = dict(event)
        visible_event["evidence_path"] = str(image_path)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(visible_event, handle, indent=2)

        return str(image_path)
