"""
NexGuard — YOLO Accident Model Training Pipeline
==================================================
Trains a YOLOv8 classification model on the CCTV accident dataset:
  - Dataset: datasets/ (Accident vs Non Accident)
  - Output model: models/accident_classifier.pt

Usage:
    python train_accident_model.py
    python train_accident_model.py --epochs 10 --batch 32 --imgsz 224
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

from config import MODELS_DIR, cfg
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.training")


def train_model(
    data_dir: Path,
    output_model_path: Path,
    epochs: int = 5,
    batch_size: int = 32,
    imgsz: int = 224,
    device: str = "cpu",
) -> Path:
    """
    Trains a YOLOv8 classification model on the accident dataset.

    Args:
        data_dir: Root dataset folder containing train/, val/, test/
        output_model_path: Destination path for the best trained model weights
        epochs: Number of training epochs
        batch_size: Mini-batch size
        imgsz: Image resolution (224x224 default for classification)
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Path to the saved best model weights.
    """
    print("=" * 60)
    print("  NEXGUARD YOLO ACCIDENT CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"  Dataset Directory : {data_dir}")
    print(f"  Epochs            : {epochs}")
    print(f"  Batch Size        : {batch_size}")
    print(f"  Image Size        : {imgsz}x{imgsz}")
    print(f"  Device            : {device}")
    print(f"  Target Output     : {output_model_path}")
    print("=" * 60)

    # Initialize YOLOv8 Nano classification model
    model = YOLO("yolov8n-cls.pt")

    # Run training
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project="models/runs",
        name="accident_cls",
        exist_ok=True,
        plots=False,
    )

    # Locate best.pt
    best_weights = Path(model.trainer.best)
    if best_weights.exists():
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, output_model_path)
        print("\n" + "=" * 60)
        print("  TRAINING COMPLETED SUCCESSFULLY")
        print(f"  Best weights saved to: {output_model_path}")
        print("=" * 60)
    else:
        print(f"[WARNING] Could not locate best weights at {best_weights}")

    # Evaluate on test set
    print("\n[INFO] Evaluating trained model on test dataset...")
    metrics = model.val(data=str(data_dir), split="test")
    top1 = metrics.top1
    top5 = metrics.top5
    print(f"\n  Test Results -> Top-1 Accuracy: {top1:.2%}, Top-5 Accuracy: {top5:.2%}\n")

    return output_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO Accident Model on CCTV footage")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS_DIR / "accident_classifier.pt"),
        help="Output model path",
    )
    args = parser.parse_args()

    dataset_path = Path(__file__).resolve().parent / "datasets"
    train_model(
        data_dir=dataset_path,
        output_model_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
