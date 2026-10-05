#!/usr/bin/env python3
"""
NexGuard — Custom Accident Model Training Script
=================================================
Trains a YOLOv8 model on a custom accident detection dataset.

NOTE: This requires a properly structured dataset.
See datasets/README.md for dataset preparation instructions.

Usage:
    python scripts/train_accident_model.py
    python scripts/train_accident_model.py --epochs 50 --batch 16 --device cuda
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a custom YOLOv8 model for accident detection"
    )
    parser.add_argument("--data", default="datasets/accident/data.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--name", default="nexguard_accident",
                        help="Experiment name for results directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found: {data_path}")
        print("Please prepare your dataset. See datasets/README.md for instructions.")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return 1

    print(f"\n{'─'*60}")
    print("  NEXGUARD — Custom Accident Model Training")
    print(f"{'─'*60}")
    print(f"  Dataset  : {args.data}")
    print(f"  Base     : {args.model}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {args.device}")
    print(f"{'─'*60}\n")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/train",
    )

    print(f"\n[INFO] Training complete. Results saved to runs/train/{args.name}")
    print("[INFO] Use the best.pt weights in your .env as NEXGUARD_MODEL_PATH")
    return 0


if __name__ == "__main__":
    sys.exit(main())
