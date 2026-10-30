"""
NexGuard — Train Accident Detection Model
=========================================
Trains and checkpoints the accident model on the validated dataset.

Usage:
    python scripts/train_object_detector.py --config configs/accident_dataset.yaml
    python scripts/train_object_detector.py --epochs 3 --batch 16 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexguard.training.trainer import AccidentModelTrainer
from nexguard.training.model_selector import ModelSelector

def main() -> int:
    parser = argparse.ArgumentParser(description="Train NexGuard Accident Model")
    parser.add_argument("--config", type=str, default="configs/accident_dataset.yaml", help="Dataset config path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image resolution (default: 224)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Load config if exists
    data_path = Path("dataset_processed")
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)
            if "path" in cfg_data:
                data_path = Path(cfg_data["path"])

    if not data_path.exists():
        data_path = Path("datasets")

    # Select model architecture
    selection = ModelSelector.select_model(task="classify", force_device=None if args.device == "auto" else args.device)
    print(f"[INFO] Model Selection: {selection.architecture} ({selection.rationale})")

    trainer = AccidentModelTrainer(
        data_dir=data_path,
        output_dir="models/object_detection",
        model_name=selection.model_name,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
    )

    record = trainer.train(resume=args.resume)
    print(f"\n[COMPLETE] Model training concluded. Best weights at: {record.best_weights_path}")
    print(f"Validation Top-1: {record.val_top1_accuracy:.2%}")
    print(f"Test Top-1:       {record.test_top1_accuracy:.2%}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
