# NexGuard Diagnostics Audit: Root Cause Analysis & Architecture Review

## 1. Executive Summary

This diagnostic document identifies the root causes of missed accident detections and pipeline pacing issues in NexGuard v2.0.0, establishing the engineering blueprint for NexGuard v2.1.0.

## 2. Model & Weights Audit

### 2.1 Loaded Weights
- **Path**: `yolov8n.pt` (6,549,796 bytes)
- **Model Type**: Generic COCO 80-class object detector (Ultralytics YOLOv8 Nano).
- **Task**: Object detection (`detect`).
- **Classes**: 80 classes including `person` (0), `bicycle` (1), `car` (2), `motorcycle` (3), `bus` (5), `truck` (7).
- **Accident-Specific Training**: **None**. There are no collision, crash, or accident classes in `yolov8n.pt`.

### 2.2 Accident Dataset & Custom Weights Status
- `datasets/accident/data.yaml`: **Missing**.
- `runs/train/nexguard_accident/weights/best.pt`: **Missing**.
- **Investigation of `NEXGUARD ERROR: ACCIDENT DATASET REQUIRED`**:
  The system currently falls back to `yolov8n.pt` without verifying whether the model is trained for accident classification. When an accident dataset or trained accident model is expected, the repository provides only documentation in `datasets/README.md` and a training script `scripts/train_accident_model.py`. Without custom accident weights, NexGuard functions purely as a **heuristic kinematic prototype** on top of generic vehicle detections.

## 3. Detection Pipeline Root Causes

### 3.1 Over-Restrictive Class Filtering in `detector.py`
In `nexguard/detection/detector.py`:
```python
target_classes = set(cfg.all_target_classes)
for box in results.boxes:
    cls_id = int(box.cls[0])
    cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
    if cls_name not in target_classes:
        continue
```
`cfg.all_target_classes` only includes COCO vehicle and person labels (`car`, `motorcycle`, `bicycle`, `bus`, `truck`, `person`). If a fine-tuned accident model is loaded with classes such as `accident`, `near_miss`, or `crash`, the detector **discards all of them**.

### 3.2 High Confidence Threshold (0.40)
The default confidence threshold in `config.py` was `0.40`. In surveillance cameras with motion blur, distant vehicles, or occlusions, vehicles are often detected at confidences between `0.20` and `0.38`. In `test_clips/traffic.mp4`, vehicles were detected at `0.36` and dropped by the 0.40 threshold.

### 3.3 Class ID Flickering in Tracking
In `nexguard/tracking/tracker.py`:
```python
if track.class_name != det.class_name:
    continue
```
Object detectors frequently alternate labels between `car`, `truck`, and `bus` across consecutive frames. This causes the tracker to drop existing track IDs and spawn new tracks, resetting track age to 1.
Because `accident_detector.py` requires `t.age >= 3`, flickering vehicles are permanently excluded from accident evaluation.

### 3.4 Premature Loop Break in Cost Matrix
In `_IoUTracker.update()`:
```python
if cost[i, j] >= 1.0:
    break
```
When `iou == 0.0` but centroid distance is within `max_dist_px`, `cost[i, j] = 1.0 - 0.0 = 1.0`. The condition `cost[i, j] >= 1.0` triggered a `break`, prematurely terminating greedy matching for all remaining detections.

### 3.5 Resolution-Independent Geometry Thresholds
In `config.py`, proximity is fixed at `120.0px` and IoU at `0.10`. In 1080p or 4K streams, vehicle centroids can be 200–300px apart during impact, meaning the proximity score remained 0.0.

### 3.6 Alert Suppression on Low Severity
In `alert_manager.py`, events classified as `Severity.LOW` (<0.55 score) are suppressed unless explicit low contacts are configured. Since heuristic collisions without pedestrian involvement often scored ~0.50, alerts were silently ignored.

## 4. Rate Limiting Audit (Hard 15 FPS Maximum)

The previous implementation used a post-loop `time.sleep(target_time - elapsed_loop)` using `time.time()`.
Defects:
1. Does not enforce rolling 1-second interval limit of 15 inference calls.
2. Does not use a monotonic clock.
3. Does not skip incoming frames on live RTSP/webcam feeds, creating unbounded queue latency.
4. Does not report actual inference FPS vs capture FPS.

## 5. Action Plan
1. Centralize `MAX_FPS = 15` in `config.py` and implement `MonotonicRateLimiter`.
2. Add honest startup diagnostics reporting model weights, classes, device, and dataset status.
3. Fix class filtering, tracking cost matrix, and vehicle class aliases.
4. Make detection confidence configurable with default 0.25.
5. Provide diagnostic mode with annotated output.
6. Improve accident event logic and severity classification.
7. Ensure non-blocking WhatsApp alerts with deduplication and cooldown.
8. Implement comprehensive unit and integration test suite.
