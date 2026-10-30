"""
NexGuard — Incident Evidence Collector
======================================
Captures forensic snapshot frames and metadata records for confirmed accidents:
- Saves annotated JPG snapshots to evidence/incidents/<incident_id>.jpg
- Writes structured JSON forensic reports with GPS/location, tracks, and reasons
- Enforces storage quotas and retention policies to protect local disk space
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from nexguard.models.schemas import Incident
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.evidence")

@dataclass
class IncidentEvidenceRecord:
    incident_id: str
    timestamp: str
    frame_index: int
    severity: str
    confidence: float
    event_type: str
    involved_tracks: List[int]
    evidence_reasons: List[str]
    snapshot_path: str
    metadata_path: str

class EvidenceCollector:
    """
    Persists forensic evidence files upon accident confirmation.
    """

    def __init__(
        self,
        evidence_dir: Path | str = "evidence/incidents",
        max_retained_incidents: int = 200,
    ) -> None:
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.max_retained = max_retained_incidents

    def capture_evidence(
        self,
        incident: Incident,
        frame: np.ndarray,
        frame_index: int,
    ) -> IncidentEvidenceRecord:
        snap_path = self.evidence_dir / f"{incident.incident_id}_frame{frame_index}.jpg"
        meta_path = self.evidence_dir / f"{incident.incident_id}_frame{frame_index}.json"

        # Save snapshot
        if frame is not None and frame.size > 0:
            cv2.imwrite(str(snap_path), frame)

        # Build metadata record
        record = IncidentEvidenceRecord(
            incident_id=incident.incident_id,
            timestamp=incident.to_dict().get("timestamp", ""),
            frame_index=frame_index,
            severity=incident.severity.value,
            confidence=round(incident.confidence, 3),
            event_type=incident.event_type.value,
            involved_tracks=incident.involved_tracks,
            evidence_reasons=incident.severity_reasons,
            snapshot_path=str(snap_path),
            metadata_path=str(meta_path),
        )

        meta_path.write_text(json.dumps(asdict(record), indent=2))
        log.info("Forensic evidence archived for incident %s -> %s", incident.incident_id, snap_path)

        self._enforce_retention_quota()
        return record

    def _enforce_retention_quota(self) -> None:
        json_files = sorted(self.evidence_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if len(json_files) > self.max_retained:
            excess = len(json_files) - self.max_retained
            for old_meta in json_files[:excess]:
                old_img = old_meta.with_suffix(".jpg")
                try:
                    old_meta.unlink(missing_ok=True)
                    old_img.unlink(missing_ok=True)
                except OSError:
                    pass
