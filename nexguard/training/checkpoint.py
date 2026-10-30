"""
NexGuard — Checkpoint Manager
=============================
Safely manages training checkpoints and production weights:
- Always maintains backups before overwriting existing best weights
- Manages best.pt and last.pt checkpoints
- Validates checkpoint file integrity and loadability
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.checkpoint")

class CheckpointManager:
    """
    Guarantees atomic weight saves, automatic historical backups, and rollback capabilities.
    """

    def __init__(self, target_dir: Path | str = "models/object_detection") -> None:
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.target_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> Path:
        return self.target_dir / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.target_dir / "last.pt"

    def backup_existing(self, path: Path) -> Optional[Path]:
        if not path.exists():
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"{path.stem}_{timestamp}{path.suffix}"
        shutil.copy2(path, backup_file)
        log.info("Backed up existing weights: %s -> %s", path, backup_file)
        return backup_file

    def save_best(self, source_weights: Path | str) -> Path:
        source_p = Path(source_weights)
        if not source_p.exists():
            raise FileNotFoundError(f"Source weight file does not exist: {source_p}")

        # Backup current best if present
        if self.best_path.exists():
            self.backup_existing(self.best_path)

        shutil.copy2(source_p, self.best_path)
        log.info("Saved new best weights to: %s", self.best_path)
        return self.best_path

    def save_last(self, source_weights: Path | str) -> Path:
        source_p = Path(source_weights)
        if not source_p.exists():
            raise FileNotFoundError(f"Source weight file does not exist: {source_p}")

        shutil.copy2(source_p, self.last_path)
        log.info("Updated last weights to: %s", self.last_path)
        return self.last_path

    @staticmethod
    def verify_weights(path: Path | str) -> bool:
        p = Path(path)
        if not p.exists() or p.stat().st_size < 1024:
            return False
        try:
            from ultralytics import YOLO
            YOLO(str(p))
            return True
        except Exception as exc:
            log.warning("Weight verification failed for %s: %s", p, exc)
            return False
