"""
NexGuard Local Incident & Frame Evidence Storage Subsystem
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from utils.config import INCIDENTS_DIR
from whatsapp.client import whatsapp_client


class IncidentRecord:
    def __init__(
        self,
        incident_id: str,
        timestamp: str,
        severity: str,
        confidence: float,
        detected_objects: Dict[str, int],
        image_path: str,
        alert_status: str,
        source_type: str = "Video"
    ):
        self.incident_id = incident_id
        self.timestamp = timestamp
        self.severity = severity
        self.confidence = confidence
        self.detected_objects = detected_objects
        self.image_path = image_path
        self.alert_status = alert_status
        self.source_type = source_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "detected_objects": self.detected_objects,
            "image_path": self.image_path,
            "alert_status": self.alert_status,
            "source_type": self.source_type
        }


class IncidentStore:
    def __init__(self, storage_dir: Path = INCIDENTS_DIR):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._incidents: Dict[str, IncidentRecord] = {}
        self._counter = 1
        self._load_existing_incidents()

    def _load_existing_incidents(self):
        meta_file = self.storage_dir / "incidents_log.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    for item in data:
                        rec = IncidentRecord(**item)
                        self._incidents[rec.incident_id] = rec
                    self._counter = len(self._incidents) + 1
            except Exception:
                pass

    def _save_index(self):
        meta_file = self.storage_dir / "incidents_log.json"
        data = [rec.to_dict() for rec in self._incidents.values()]
        with open(meta_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_incident(
        self,
        frame: np.ndarray,
        severity_level: str,
        confidence: float,
        detected_objects: Dict[str, int],
        source_type: str = "CCTV"
    ) -> IncidentRecord:
        date_str = datetime.now().strftime("%Y%m%d")
        incident_id = f"NG-{date_str}-{self._counter:04d}"
        self._counter += 1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save evidence frame image
        img_filename = f"{incident_id}.jpg"
        img_path = self.storage_dir / img_filename
        cv2.imwrite(str(img_path), frame)

        # Build detected summary string
        summary_parts = [f"{cnt} {cls.capitalize()}s" if cnt > 1 else f"1 {cls.capitalize()}" for cls, cnt in detected_objects.items()]
        summary_str = ", ".join(summary_parts) if summary_parts else "Vehicles involved"

        # Attempt WhatsApp alert (local QR authentication checks inside)
        alert_res = whatsapp_client.send_accident_alert(
            incident_id=incident_id,
            severity_level=severity_level,
            confidence=confidence,
            detected_summary=summary_str,
            timestamp=timestamp,
            source=source_type
        )

        record = IncidentRecord(
            incident_id=incident_id,
            timestamp=timestamp,
            severity=severity_level,
            confidence=confidence,
            detected_objects=detected_objects,
            image_path=str(img_path),
            alert_status=alert_res.get("status", "NOT SENT"),
            source_type=source_type
        )

        self._incidents[incident_id] = record
        self._save_index()
        return record

    def list_incidents(self) -> List[Dict[str, Any]]:
        return [rec.to_dict() for rec in reversed(list(self._incidents.values()))]


incident_store = IncidentStore()
