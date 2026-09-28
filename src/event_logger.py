"""
NexGuard Event Logger Module
Logs detection telemetry and event summaries to structured JSON records.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from src.logger import get_logger

logger = get_logger()


class DetectionEventLogger:
    """Logs structured telemetry events to JSON lines file."""

    def __init__(self, log_path: str = "logs/detection_events.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_detection(self, frame_id: int, detections: List[Dict[str, Any]], fps: float) -> None:
        """Writes a single frame detection event log."""
        if not detections:
            return

        class_counts: Dict[str, int] = {}
        for d in detections:
            cname = d.get("class_name", "unknown")
            class_counts[cname] = class_counts.get(cname, 0) + 1

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": time.time(),
            "frame_id": frame_id,
            "fps": round(fps, 1),
            "total_objects": len(detections),
            "counts": class_counts
        }

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write detection event log: {e}")
