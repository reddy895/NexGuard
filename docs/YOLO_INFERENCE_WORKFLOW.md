# NexGuard — YOLO Inference Pipeline Workflow

This document details the architectural design and data flow of the YOLO object detection engine in **NexGuard Edge AI Surveillance System**.

---

## 🏗️ Detection Data Flow Architecture

```text
  ┌────────────────────────┐
  │  Input Source Stream   │
  │ (Webcam/Video/Image)   │
  └───────────┬────────────┘
              │ 1. Read Raw Frame (BGR ndarray)
              ▼
  ┌────────────────────────┐
  │    NexGuardDetector    │
  │ 2. Device Acceleration │  <── Auto-detect CUDA vs CPU
  │ 3. YOLO Inference      │  <── Model: yolov8n.pt
  │ 4. Threshold Filter    │  <── Conf >= 0.25, IoU >= 0.45
  └───────────┬────────────┘
              │ 5. Detection Dictionaries List
              ▼
  ┌────────────────────────┐
  │   Display Visualizer   │
  │ 6. Class BBoxes & Text │  <── Dynamic COCO Class Colors
  │ 7. Telemetry Overlay   │  <── Live FPS, Frames, Objects
  └───────────┬────────────┘
              │ 8. Render Frame / User Keypress
              ▼
  ┌────────────────────────┐
  │   OpenCV Window / CLI  │  <── Q (Quit), P (Pause), S (Snapshot), R (Reset)
  └────────────────────────┘
```

---

## 🔍 Core Inference Pipeline Breakdown

### 1. Model Loading & Hardware Detection (`src/detector.py`)
- Hardware acceleration is auto-detected on startup via `src.utils.detect_device()`.
- If CUDA hardware is present (NVIDIA GPU), PyTorch loads models directly onto CUDA VRAM.
- Otherwise, inference falls back to CPU execution without breaking application flow.

### 2. Confidence & IoU Filtering
- Objects detected by YOLO are evaluated against `confidence_threshold` (default `0.25`).
- Non-Maximum Suppression (NMS) removes duplicate overlapping bounding boxes using `iou_threshold` (default `0.45`).

### 3. Dynamic Class Mapping
- Class names are read dynamically from model metadata (`model.names`).
- Typical COCO surveillance objects include:
  - `person` (ID 0)
  - `car` (ID 2)
  - `motorcycle` (ID 3)
  - `bus` (ID 5)
  - `truck` (ID 7)

### 4. Evidence Snapshot Mechanism
- Pressing `S` captures the current frame with bounding box overlays and saves it directly to `outputs/detection_YYYYMMDD_HHMMSS.jpg`.

---

*NexGuard Edge AI Technical Documentation*
