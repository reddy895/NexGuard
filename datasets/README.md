# NexGuard — Accident Detection Dataset & Training Guide

## Overview

This directory holds accident detection and classification training data from CCTV footage.
The system integrates with YOLOv8 to train on real-world road accident and non-accident scenarios.

## Current Dataset: CCTV Accident Detection

Downloaded and configured from Kaggle:
`ckay16/accident-detection-from-cctv-footage`

### Structure

```
datasets/
├── README.md               ← This documentation
├── train/
│   ├── Accident/           ← 369 training images
│   └── Non Accident/       ← 422 training images
├── val/
│   ├── Accident/           ← 46 validation images
│   └── Non Accident/       ← 52 validation images
└── test/
    ├── Accident/           ← 47 test images
    └── Non Accident/       ← 54 test images
```

Total: **990 CCTV images** across `Accident` and `Non Accident` categories.

---

## Model Training

A dedicated training script is available in the root directory: [`train_accident_model.py`](file:///home/reddy895/Downloads/NexGuard/train_accident_model.py).

### Run Training

```bash
python train_accident_model.py --epochs 5 --batch 32 --imgsz 224
```

### Options:
- `--epochs`: Number of training epochs (default: `5`).
- `--batch`: Batch size (default: `32`).
- `--imgsz`: Input image resolution (default: `224`).
- `--device`: Target device (`cpu` or `cuda`).
- `--output`: Output model destination (default: `models/accident_classifier.pt`).

### Results:
- Top-1 Validation Accuracy: **~87.8%**
- Best weights automatically saved to `models/accident_classifier.pt`.

---

## Testing the Model

Run inference on test images:

```python
from ultralytics import YOLO

model = YOLO("models/accident_classifier.pt")
results = model("datasets/test/Accident/test5_12.jpg")
print(results[0].probs.top1)  # Class index
```

---

## Git Guidelines

Large media files (`train/`, `val/`, `test/`, and `*.cache`) are automatically ignored via `.gitignore` to keep the repository lightweight.
