"""
NexGuard — Temporal Feature Aggregator
======================================
Aggregates kinematic, spatial, and visual features across rolling temporal windows:
- Measures multi-frame collision persistence
- Extracts concurrent deceleration and post-impact standstill duration
- Prevents single-frame transient false alarms
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional
import numpy as np
from nexguard.models.schemas import Track

@dataclass
class TemporalAccidentFeatures:
    pair_id: str
    max_iou: float
    min_dist_px: float
    max_approach_vel: float
    max_deceleration_ratio: float
    post_impact_standstill_frames: int
    consecutive_overlap_frames: int
    mean_model_confidence: float

class TemporalFeatureAggregator:
    """
    Maintains temporal buffers across track pairs to compute composite accident features.
    """

    def __init__(self, window_size: int = 15) -> None:
        self.window_size = window_size
        self._history: Dict[str, Deque[Dict[str, float]]] = {}

    @staticmethod
    def get_pair_key(t1_id: int, t2_id: int) -> str:
        return f"{min(t1_id, t2_id)}_{max(t1_id, t2_id)}"

    def record_interaction(
        self,
        t1: Track,
        t2: Track,
        iou: float,
        dist: float,
        approach_vel: float,
        model_conf: float = 0.0,
    ) -> None:
        key = self.get_pair_key(t1.track_id, t2.track_id)
        if key not in self._history:
            self._history[key] = deque(maxlen=self.window_size)

        max_drop = max(t1.speed_drop_ratio, t2.speed_drop_ratio)
        is_standstill = (t1.speed < 0.5 or t2.speed < 0.5) and iou > 0.05

        self._history[key].append({
            "iou": iou,
            "dist": dist,
            "approach_vel": approach_vel,
            "decel": max_drop,
            "standstill": 1.0 if is_standstill else 0.0,
            "model_conf": model_conf,
        })

    def extract_features(self, t1_id: int, t2_id: int) -> Optional[TemporalAccidentFeatures]:
        key = self.get_pair_key(t1_id, t2_id)
        buf = self._history.get(key)
        if not buf or len(buf) < 2:
            return None

        ious = [frame["iou"] for frame in buf]
        dists = [frame["dist"] for frame in buf]
        approaches = [frame["approach_vel"] for frame in buf]
        decels = [frame["decel"] for frame in buf]
        standstills = [frame["standstill"] for frame in buf]
        model_confs = [frame["model_conf"] for frame in buf]

        consec_overlap = sum(1 for iou in ious if iou > 0.05)
        standstill_count = int(sum(standstills))

        return TemporalAccidentFeatures(
            pair_id=key,
            max_iou=max(ious),
            min_dist_px=min(dists),
            max_approach_vel=max(approaches),
            max_deceleration_ratio=max(decels),
            post_impact_standstill_frames=standstill_count,
            consecutive_overlap_frames=consec_overlap,
            mean_model_confidence=float(np.mean(model_confs)),
        )

    def prune_stale(self, active_track_ids: List[int]) -> None:
        active_set = set(active_track_ids)
        to_del = []
        for key in self._history:
            parts = key.split("_")
            id1, id2 = int(parts[0]), int(parts[1])
            if id1 not in active_set and id2 not in active_set:
                to_del.append(key)
        for k in to_del:
            del self._history[k]
