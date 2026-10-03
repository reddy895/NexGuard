"""
NexGuard Custom Accident YOLO Validation Script
Validates custom accident model performance metrics against validation dataset.
"""

from pathlib import Path
import torch
from ultralytics import YOLO

from config import config, BASE_DIR, DATASET_DIR, CUSTOM_MODEL_DIR
from utils.validation import validate_accident_dataset


def validate_accident_model() -> bool:
    """Validates model weights against dataset validation split."""
    custom_model_file = Path(config.custom_model_path)
    runs_model_file = BASE_DIR / "runs" / "accident" / "accident_detector" / "weights" / "best.pt"

    target_model = None
    if custom_model_file.exists():
        target_model = custom_model_file
    elif runs_model_file.exists():
        target_model = runs_model_file

    if not target_model:
        print("\n==================================================")
        print("NEXGUARD ACCIDENT MODEL VALIDATION ERROR")
        print("==================================================")
        print(f"No custom accident model found at: {config.custom_model_path}")
        print("Train a custom model first using option 5 or supply models/custom/accident.pt")
        print("==================================================\n")
        return False

    is_valid, msg = validate_accident_dataset(DATASET_DIR)
    if not is_valid:
        print("\n==================================================")
        print("NEXGUARD ACCIDENT MODEL VALIDATION ERROR")
        print("==================================================")
        print(f"Validation dataset missing or invalid: {msg}")
        print("==================================================\n")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_yaml = DATASET_DIR / "data.yaml"

    print("\n==================================================")
    print("NEXGUARD ACCIDENT MODEL VALIDATION")
    print("==================================================")
    print(f"Evaluating Model : {target_model}")
    print(f"Dataset Config   : {data_yaml}")
    print(f"Device           : {device.upper()}")
    print("==================================================\n")

    try:
        model = YOLO(str(target_model))
        metrics = model.val(data=str(data_yaml), device=device)

        print("\n==================================================")
        print("VALIDATION METRICS SUMMARY")
        print("==================================================")
        print(f"Precision (P)   : {metrics.box.map50:.4f}")
        print(f"Recall (R)      : {metrics.box.map:.4f}")
        print(f"mAP@50          : {metrics.box.map50:.4f}")
        print(f"mAP@50-95       : {metrics.box.map:.4f}")
        print("==================================================\n")
        return True
    except Exception as e:
        print(f"\nNEXGUARD VALIDATION ERROR: {e}\n")
        return False


if __name__ == "__main__":
    validate_accident_model()
