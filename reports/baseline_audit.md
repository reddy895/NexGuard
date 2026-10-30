# NexGuard — Repository Audit & Baseline Architecture Report

**Date:** 2026-10-30  
**Repository:** NexGuard (AI Road Accident Detection & Emergency Response)  
**Base Commit:** `728a606`

---

## 1. System & Hardware Audit

- **Operating System:** Linux x86_64 (Kernel 6.17)
- **CPU:** Multi-core Intel/AMD host
- **GPU:** NVIDIA GeForce RTX 2050 (4096 MiB VRAM, Driver 595.99.02, CUDA 13.2 capability)
- **Current PyTorch Environment:** PyTorch 2.14.1+cpu (Host environment managed via Debian PEP 668, CPU execution active with GPU fallback logic implemented)
- **Ultralytics Version:** 8.4.173
- **OpenCV Version:** 5.0.0.93

---

## 2. Repository Layout & Existing Modules

1. **`nexguard/detection/detector.py`**:
   - `ObjectDetector` wraps Ultralytics YOLO (`yolov8n.pt`).
   - Supports vehicle (`car`, `motorcycle`, `bus`, `truck`, `bicycle`) and person detection.
   - Filters classes to configured target set.
   - Diagnostic info capturing model status and device selection.

2. **`nexguard/tracking/tracker.py`**:
   - `ObjectTracker` implements centroid & spatial distance tracking with unique ID assignment, velocity calculation, and motion history.

3. **`nexguard/accident/accident_detector.py`**:
   - Multi-signal heuristic detector analyzing:
     - IoU bounding-box overlap.
     - Centroid proximity with adaptive bounding box scale.
     - Approaching (closing) velocity.
     - Sudden deceleration / speed drops.
     - Heading divergence after contact.
     - Pedestrian proximity to vehicle collision zone.
     - Sudden stop analysis for stationary obstacles.
     - Direct model detection confirmation hook.

4. **`nexguard/accident/severity.py`**:
   - `SeverityClassifier` computing a transparent weighted composite score [0.0 - 1.0] across collision intensity, speed drop, vehicle count, pedestrian involvement, event persistence, and scene density.
   - Categorizes incidents into `LOW`, `MEDIUM`, `HIGH`, or `UNCERTAIN`.

5. **`nexguard/accident/event.py`**:
   - `IncidentManager` managing active incidents, deduplication by track IDs, persistence frames, stale resolution, and history bounds.

6. **`video_processor.py`**:
   - Real-time video frame processor supporting file feeds and webcams, overlay rendering, and continuous looping.

7. **`tests/`**:
   - 46 unit/integration tests covering detector pipelines, tracker, accident confirmation, severity classification, and error handling. Baseline run: **46 passed in 24.08s**.

---

## 3. Existing Model Artifacts

- `yolov8n.pt`: 6.55 MB (Base COCO pretrained object detector)
- `yolov8n-cls.pt`: 5.56 MB (Base ImageNet pretrained classification backbone)
- `models/accident_classifier.pt`: 5.6 MB (Preliminary CCTV classification weights)

---

## 4. Initial Findings & Gaps

1. **Dataset Format**: The dataset in `datasets/` consists of image classification folders (`Accident/` and `Non Accident/`), **not** bounding-box object detection annotations.
2. **Data Leakage**: Sequential frames from identical video sources are split randomly across `train/`, `val/`, and `test/`.
3. **Duplicate Images & Label Conflicts**: Near-duplicate and exact MD5 duplicate images exist between train and val with conflicting class labels.
4. **Hardware Adaptation**: High-resolution training requires conservative batch sizing and mixed precision/CPU fallback resilience to guarantee stable execution on 4GB VRAM or CPU.
