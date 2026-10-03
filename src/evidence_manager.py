"""
NexGuard Evidence Capture & File Management Module
Saves evidence frames and incident metadata to the local incidents/ directory.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from config import config, INCIDENTS_DIR
from utils.logger import logger


class EvidenceManager:
    """Manages saving image evidence frames and structured JSON incident reports."""

    def __init__(self, target_dir: Path = INCIDENTS_DIR):
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def save_evidence(
        self,
        frame: np.ndarray,
        incident_id: str,
        metadata: Dict[str, Any]
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Saves frame as JPEG and metadata as JSON.
        Returns (image_path, json_path).
        """
        if frame is None:
            return None, None

        timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_prefix = f"incident_{timestamp_str}_{incident_id}"

        img_path = self.target_dir / f"{file_prefix}.jpg"
        json_path = self.target_dir / f"{file_prefix}.json"

        try:
            cv2.imwrite(str(img_path), frame)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved incident evidence to: {img_path}")
            return img_path, json_path
        except Exception as e:
            logger.error(f"Failed to save evidence for incident {incident_id}: {e}")
            return None, None
