# NexGuard — Accident Detection Dataset

## Overview

This directory holds custom accident detection training data.
The system uses YOLOv8 object detection format (COCO-style bounding boxes).

## Expected Structure

```
datasets/
    accident/
        images/
            train/      ← Training images (.jpg/.png)
            val/        ← Validation images
            test/       ← Test images
        labels/
            train/      ← Annotation .txt files (YOLO format)
            val/
            test/
        data.yaml       ← Dataset configuration
```

## data.yaml Format

```yaml
path: datasets/accident
train: images/train
val: images/val
test: images/test

nc: 2  # number of classes
names:
  0: accident
  1: near_miss
```

## Annotation Format (YOLO)

Each `.txt` label file corresponds to an image.
Each line describes one bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0.0–1.0) relative to image dimensions.

Example:
```
0 0.512 0.435 0.340 0.280
```

## Recommended Datasets

| Dataset | URL |
|---------|-----|
| DOTA (traffic) | https://captain-whu.github.io/DOTA/ |
| AI City Challenge | https://www.aicitychallenge.org/ |
| CCD (Car Crash) | https://github.com/Cogito2012/CarCrashDataset |
| CADP | https://ankitshah009.github.io/accident_forecasting_traffic_camera |

## Training Command

```bash
python scripts/train_accident_model.py \
    --data datasets/accident/data.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --device cuda
```

## After Training

The best model weights will be at:
```
runs/train/nexguard_accident/weights/best.pt
```

Set this in your `.env`:
```
NEXGUARD_MODEL_PATH=runs/train/nexguard_accident/weights/best.pt
```

## Important Note

The current NexGuard accident detection layer is a **heuristic prototype**
built on top of general object detection (YOLO COCO model).
Training a custom accident detection model will significantly improve accuracy.

Do NOT commit large datasets to Git. Use `.gitignore` to exclude image and label directories.
