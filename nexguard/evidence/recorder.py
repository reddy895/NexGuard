"""
NexGuard — Evidence Recorder
==============================
Captures and organizes evidence for confirmed accident incidents.

Evidence structure:
    evidence/incidents/
        YYYY-MM-DD/
            INC-XXXXXXXX/
                frame_001.jpg
                frame_002.jpg
                metadata.json
                clip.avi         (if video recording enabled)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import cfg
from nexguard.models.schemas import Incident
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.evidence")


class EvidenceRecorder:
    """
    Stores evidence frames, metadata, and optional short video clips
    for confirmed accident incidents.

    Evidence is organized by date and incident ID under the configured
    evidence directory. Retention enforcement removes old evidence
    based on the configured retention period.
    """

    def __init__(
        self,
        evidence_dir: Optional[str] = None,
        retention_days: Optional[int] = None,
        max_frames_per_incident: int = 10,
    ) -> None:
        self._base = Path(evidence_dir or cfg.evidence_dir)
        self._retention_days = retention_days or cfg.evidence_retention_days
        self._max_frames = max_frames_per_incident
        self._base.mkdir(parents=True, exist_ok=True)

        # Active video writers: incident_id → cv2.VideoWriter
        self._video_writers: dict = {}

    def save_frame(
        self,
        incident: Incident,
        frame: np.ndarray,
        frame_index: int,
    ) -> Optional[str]:
        """
        Saves a single evidence frame for an incident.

        Returns the saved file path, or None if the limit was reached.
        """
        if len(incident.evidence_frames) >= self._max_frames:
            return None

        incident_dir = self._incident_dir(incident)
        incident_dir.mkdir(parents=True, exist_ok=True)

        filename = f"frame_{frame_index:06d}.jpg"
        path = incident_dir / filename

        try:
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            log.debug(f"Evidence frame saved: {path}")
            return str(path)
        except Exception as e:
            log.warning(f"Failed to save evidence frame: {e}")
            return None

    def save_metadata(self, incident: Incident) -> Optional[str]:
        """
        Saves incident metadata as a JSON file.

        Returns the path to the saved metadata file.
        """
        incident_dir = self._incident_dir(incident)
        incident_dir.mkdir(parents=True, exist_ok=True)

        path = incident_dir / "metadata.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(incident.to_dict(), f, indent=2, default=str)
            log.info(f"Evidence metadata saved: {path}")
            return str(path)
        except Exception as e:
            log.warning(f"Failed to save metadata: {e}")
            return None

    def start_clip(
        self,
        incident: Incident,
        frame: np.ndarray,
        fps: float = 15.0,
    ) -> bool:
        """
        Starts recording a short video clip for the incident.

        Returns True if the writer was initialized successfully.
        """
        if incident.incident_id in self._video_writers:
            return True

        incident_dir = self._incident_dir(incident)
        incident_dir.mkdir(parents=True, exist_ok=True)

        h, w = frame.shape[:2]
        path = incident_dir / "clip.avi"

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

        if not writer.isOpened():
            log.warning(f"Cannot open video writer for {path}")
            return False

        self._video_writers[incident.incident_id] = {
            "writer": writer,
            "path": str(path),
            "frame_count": 0,
            "max_frames": int(fps * 10),  # 10 seconds max
        }
        log.info(f"Evidence clip recording started: {path}")
        return True

    def write_clip_frame(self, incident: Incident, frame: np.ndarray) -> None:
        """Appends a frame to the incident's video clip."""
        entry = self._video_writers.get(incident.incident_id)
        if entry is None:
            return
        if entry["frame_count"] >= entry["max_frames"]:
            self.finalize_clip(incident)
            return
        entry["writer"].write(frame)
        entry["frame_count"] += 1

    def finalize_clip(self, incident: Incident) -> Optional[str]:
        """
        Closes the video writer for this incident.

        Returns:
            Path to the saved clip, or None.
        """
        entry = self._video_writers.pop(incident.incident_id, None)
        if entry is None:
            return None
        entry["writer"].release()
        log.info(
            f"Evidence clip finalized: {entry['path']} "
            f"({entry['frame_count']} frames)"
        )
        return entry["path"]

    def finalize_all(self) -> None:
        """Closes all open video writers (call on shutdown)."""
        for iid in list(self._video_writers.keys()):
            entry = self._video_writers.pop(iid)
            entry["writer"].release()

    def enforce_retention(self) -> int:
        """
        Removes evidence directories older than the retention period.

        Returns:
            Number of directories removed.
        """
        cutoff = datetime.now() - timedelta(days=self._retention_days)
        removed = 0

        for date_dir in self._base.iterdir():
            if not date_dir.is_dir():
                continue
            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff:
                    shutil.rmtree(date_dir)
                    removed += 1
                    log.info(f"Removed old evidence: {date_dir}")
            except ValueError:
                pass  # Directory not a date folder — skip

        if removed:
            log.info(f"Retention enforcement: removed {removed} old evidence directories.")
        return removed

    def _incident_dir(self, incident: Incident) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self._base / date_str / incident.incident_id
