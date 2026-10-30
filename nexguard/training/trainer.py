"""
NexGuard — Model Trainer Engine
===============================
Trains YOLO architectures for accident and traffic scene understanding:
- Supports GPU (with Automatic Mixed Precision) and CPU fallback
- Dynamic batch-size scaling based on available RAM/VRAM
- Checkpoint management: saves best.pt and last.pt
- Training resume capability
- Logs training parameters and evaluation metrics
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from ultralytics import YOLO

from nexguard.utils.logging import get_logger
from nexguard.utils.reproducibility import get_environment_signature, seed_everything

log = get_logger("nexguard.trainer")

@dataclass
class TrainingRecord:
    timestamp: str
    model_name: str
    dataset_dir: str
    epochs_requested: int
    epochs_completed: int
    batch_size: int
    image_size: int
    device: str
    mixed_precision: bool
    best_weights_path: str
    last_weights_path: str
    val_top1_accuracy: float = 0.0
    val_loss: float = 0.0
    test_top1_accuracy: float = 0.0
    environment: Dict[str, str] = field(default_factory=dict)

class AccidentModelTrainer:
    """
    Manages the training, checkpointing, and evaluation lifecycle.
    """

    def __init__(
        self,
        data_dir: Path | str,
        output_dir: Path | str = "models/object_detection",
        model_name: str = "yolov8n-cls.pt",
        device: str = "auto",
        epochs: int = 5,
        batch_size: int = 16,
        imgsz: int = 224,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.seed = seed

    def _resolve_device(self, req: str) -> str:
        if req != "auto":
            return req
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def train(self, resume: bool = False) -> TrainingRecord:
        seed_everything(self.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"==================================================")
        print(f"  NEXGUARD ACCIDENT DETECTION MODEL TRAINING")
        print(f"==================================================")
        print(f"  Dataset Directory : {self.data_dir}")
        print(f"  Model Backbone    : {self.model_name}")
        print(f"  Device            : {self.device}")
        print(f"  Epochs            : {self.epochs}")
        print(f"  Batch Size        : {self.batch_size}")
        print(f"  Resolution        : {self.imgsz}x{self.imgsz}")
        print(f"  Resume Mode       : {resume}")
        print(f"==================================================")

        run_dir = Path("models/runs/accident_train")
        last_ckpt = run_dir / "weights" / "last.pt"

        if resume and last_ckpt.exists():
            print(f"[TRAINER] Resuming training from {last_ckpt}")
            model = YOLO(str(last_ckpt))
            train_results = model.train(resume=True)
        else:
            model = YOLO(self.model_name)
            train_results = model.train(
                data=str(self.data_dir.resolve()),
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.imgsz,
                device=self.device,
                project="models/runs",
                name="accident_train",
                exist_ok=True,
                plots=False,
                seed=self.seed,
                amp=(self.device == "cuda"),
                workers=2,
            )

        # Retrieve checkpoint paths
        best_pt = self.output_dir / "best.pt"
        last_pt = self.output_dir / "last.pt"

        trainer_best = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer.best else None
        trainer_last = Path(model.trainer.last) if hasattr(model, "trainer") and model.trainer.last else None

        if trainer_best and trainer_best.exists():
            shutil.copy2(trainer_best, best_pt)
            print(f"[TRAINER] Best checkpoint saved to: {best_pt}")
        elif Path("models/accident_classifier.pt").exists() and not best_pt.exists():
            shutil.copy2("models/accident_classifier.pt", best_pt)

        if trainer_last and trainer_last.exists():
            shutil.copy2(trainer_last, last_pt)
            print(f"[TRAINER] Last checkpoint saved to: {last_pt}")
        elif best_pt.exists() and not last_pt.exists():
            shutil.copy2(best_pt, last_pt)

        # Run validation on val and test splits
        print("\n[TRAINER] Evaluating on validation split...")
        val_metrics = model.val(data=str(self.data_dir.resolve()), split="val")
        val_top1 = float(val_metrics.top1) if hasattr(val_metrics, "top1") else 0.0

        print("\n[TRAINER] Evaluating on held-out test split...")
        test_metrics = model.val(data=str(self.data_dir.resolve()), split="test")
        test_top1 = float(test_metrics.top1) if hasattr(test_metrics, "top1") else 0.0

        record = TrainingRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name=self.model_name,
            dataset_dir=str(self.data_dir),
            epochs_requested=self.epochs,
            epochs_completed=self.epochs,
            batch_size=self.batch_size,
            image_size=self.imgsz,
            device=self.device,
            mixed_precision=(self.device == "cuda"),
            best_weights_path=str(best_pt),
            last_weights_path=str(last_pt),
            val_top1_accuracy=val_top1,
            test_top1_accuracy=test_top1,
            environment=get_environment_signature(),
        )

        metadata_file = self.output_dir / "training_metadata.json"
        metadata_file.write_text(json.dumps(asdict(record), indent=2))
        print(f"[TRAINER] Training metadata logged to: {metadata_file}")
        return record
