"""
NexGuard Custom Accident YOLO Training Pipeline
Checks dataset structure and fine-tunes custom accident YOLO model.
"""

import sys
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

from config import config, BASE_DIR, DATASET_DIR, CUSTOM_MODEL_DIR, BASE_MODEL_DIR
from utils.validation import validate_accident_dataset
from utils.logger import logger


def train_accident_model(epochs: int = 50, imgsz: int = 640, batch: int = 16) -> bool:
    """Validates dataset and launches YOLO accident fine-tuning if legitimate dataset is present."""
    is_valid, msg = validate_accident_dataset(DATASET_DIR)
    if not is_valid:
        print("\n==================================================")
        print("NEXGUARD ACCIDENT MODEL TRAINING ERROR")
        print("==================================================")
        print(f"Reason: {msg}")
        print("\nExpected directory structure:")
        print("training/dataset/")
        print("    ├── images/")
        print("    │   ├── train/")
        print("    │   ├── val/")
        print("    │   └── test/")
        print("    ├── labels/")
        print("    │   ├── train/")
        print("    │   ├── val/")
        print("    │   └── test/")
        print("    └── data.yaml")
        print("==================================================\n")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_file = Path(config.base_model_path)
    if not base_model_file.exists():
        base_model_file = BASE_DIR / "yolov8n.pt"

    data_yaml = DATASET_DIR / "data.yaml"

    print("\n==================================================")
    print("NEXGUARD CUSTOM ACCIDENT YOLO TRAINING")
    print("==================================================")
    print(f"Device        : {device.upper()}")
    print(f"Base Model    : {base_model_file}")
    print(f"Dataset Config: {data_yaml}")
    print(f"Target Epochs : {epochs}")
    print("==================================================\n")

    try:
        model = YOLO(str(base_model_file) if base_model_file.exists() else "yolov8n.pt")
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(BASE_DIR / "runs" / "accident"),
            name="accident_detector",
            exist_ok=True,
            verbose=True
        )

        best_weights = BASE_DIR / "runs" / "accident" / "accident_detector" / "weights" / "best.pt"
        if best_weights.exists():
            target_path = CUSTOM_MODEL_DIR / "accident.pt"
            shutil.copy(best_weights, target_path)
            print("\n==================================================")
            print("TRAINING COMPLETED SUCCESSFULLY")
            print(f"BEST ACCIDENT MODEL SAVED TO: {target_path}")
            print("==================================================\n")
            return True
        else:
            print("\nNEXGUARD TRAINING ERROR: Trained weights file not found.\n")
            return False
    except Exception as e:
        print(f"\nNEXGUARD TRAINING ERROR: {e}\n")
        return False


if __name__ == "__main__":
    train_accident_model()
