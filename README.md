# NexGuard — Local AI Road Accident Detection System

<p align="center">
  <strong>High-performance, local terminal-based computer vision system for road accident detection & severity assessment.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Computer%20Vision-7C3AED?style=for-the-badge&logo=opencv&logoColor=white" alt="Computer Vision" />
  <img src="https://img.shields.io/badge/Object%20Detection-YOLOv8-00FFFF?style=for-the-badge&logoColor=black" alt="YOLOv8 Object Detection" />
  <img src="https://img.shields.io/badge/Language-Python%203.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Execution-Terminal%20CLI-success?style=for-the-badge" alt="Terminal Execution" />
  <img src="https://img.shields.io/badge/Testing-45%2F45%20Passed-brightgreen?style=for-the-badge" alt="Tests" />
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Exact User Experience](#exact-user-experience)
- [Architecture & Modules](#architecture--modules)
- [Detection & Severity Logic](#detection--severity-logic)
- [Installation & Setup](#installation--setup)
- [Running the System](#running-the-system)
- [Testing](#testing)
- [What Works Now vs Future Requirements](#what-works-now-vs-future-requirements)
- [Known Limitations & Troubleshooting](#known-limitations--troubleshooting)
- [License](#license)

---

## Overview

**NexGuard** is a local, terminal-based road accident detection system designed for CCTV surveillance and real-time video processing. It runs entirely on the host machine without external web servers, cloud databases, paid APIs, or mandatory internet connectivity.

The system performs:
1. **Object Detection**: Fast YOLOv8 inference detecting vehicles (`car`, `motorcycle`, `bicycle`, `bus`, `truck`) and vulnerable road users (`person`).
2. **Multi-Object Tracking**: Centroid- and IoU-based persistent object tracking with velocity and trajectory estimation.
3. **Temporal Accident Detection**: Multi-frame collision confirmation based on approach velocity, trajectory intersections, spatial proximity, and deceleration patterns.
4. **Observable Severity Classification**: Rule-based visual classification into `LOW`, `MEDIUM`, `HIGH`, or `UNCERTAIN` based strictly on physical evidence.
5. **Real-time Visualization**: Responsive OpenCV overlay with distinct class bounding boxes, persistent incident banners, telemetry metrics, and high-severity red highlights.

---

## Key Features

- **100% Local Inference**: Zero telemetry, zero cloud dependencies, zero external network requirements.
- **Terminal-Driven Workflow**: Interactive terminal menu launched via `python main.py`.
- **System Camera & File Processing**: Live capture via default webcam (device 0) or local video file processing using the OS native file dialog.
- **Color-Coded Class Visualizations**:
  - Cars: Blue (`#00A5FF`)
  - Motorcycles & Bicycles: Amber/Yellow (`#FFA500`)
  - People: Magenta (`#E040FB`)
  - Trucks & Buses: Teal (`#00E5FF`)
  - Incidents: Visual severity treatment (Amber for Low, Orange for Medium, Red for High).
- **Incident Lifecycle & Memory Cap**: Bounded memory footprint (capped history) with automatic incident deduplication and resolution when objects exit the frame.
- **Graceful Error Handling**: Fallbacks for missing model weights, missing cameras, unsupported video formats, and smooth return to the menu on `Q` or `ESC`.

---

## Exact User Experience

Running `python main.py` greets the user with an interactive terminal interface:

```text
==================================================
                 NEXGUARD AI
       ROAD ACCIDENT DETECTION SYSTEM
==================================================

Select an operation:

[1] Live Detection — System Camera
[2] Detect Accident in a Video File
[3] Exit

Enter your choice: 
```

### Option 1: Live Detection (Webcam)
- Connects to the primary system webcam (`device 0`).
- Runs real-time frame capture and overlay rendering in an OpenCV window.
- Terminal outputs frame progress and detected incidents.
- Pressing `Q` or `ESC` closes the window, safely releases the camera, and cleanly returns to the terminal menu.

### Option 2: Detect Accident in a Video File
- Launches the OS native file picker (Tkinter dialog) to select an MP4, AVI, MOV, MKV, or WEBM file.
- Plays and analyzes video frames sequentially, displaying metrics (FPS, frame counter, detected vehicles, persons).
- Automatically returns to the menu when video finishes or when `Q` is pressed.

### Option 3: Exit
- Releases all active resources, terminates the process, and returns cleanly to the shell.

---

## Architecture & Modules

```text
NexGuard/
├── main.py                 # Terminal menu & application lifecycle entry point
├── config.py               # Centralized configuration (thresholds, device, classes)
├── video_processor.py      # Video capture loop for webcam and file sources
├── overlay.py              # HUD overlay, telemetry, and bounding box renderer
├── requirements.txt        # Minimal Python dependencies
├── README.md               # Documentation and execution instructions
├── nexguard/
│   ├── detection/
│   │   └── detector.py     # YOLOv8 model inference & class mapping
│   ├── tracking/
│   │   └── tracker.py      # IoU & distance multi-object tracking
│   ├── accident/
│   │   ├── accident_detector.py # Temporal collision detection engine
│   │   ├── severity.py     # SeverityClassifier & person involvement analyzer
│   │   └── event.py        # IncidentManager lifecycle & deduplication
│   ├── models/
│   │   └── schemas.py      # Dataclasses & Enums (Severity, Detection, Track, etc.)
│   └── utils/
│       ├── logging.py      # Centralized logging configuration
│       └── utils.py        # Geometric math (IoU, distance, heading) & FPS counter
└── tests/
    ├── test_severity.py    # 14 unit tests for severity state transitions
    ├── test_accident_logic.py # 18 unit tests for accident logic & cooldown
    └── test_system_and_errors.py # 13 unit tests for pipeline errors & edge cases
```

---

## Detection & Severity Logic

### Accident Event Confirmation
NexGuard prevents false positives from single-frame overlaps by enforcing **temporal confirmation**:
- Potential collisions must satisfy proximity, approach velocity, or IoU overlap across consecutive frames (`accident_confirm_frames = 4`).
- While under evaluation, the overlay displays `POSSIBLE ACCIDENT`.
- Upon confirmation, an `Incident` is generated with a unique ID (e.g. `NG-0001`).
- A cooldown window (`accident_cooldown_frames = 60`) prevents duplicate alerts for the same collision.

### Severity Classification Rules
The `SeverityClassifier` assigns an estimated continuous score `[0.0, 1.0]` based on measurable computer vision signals:
1. **Collision Intensity**: Detection confidence of the collision event.
2. **Kinematic Deceleration**: Post-impact speed drop ratio across involved vehicles.
3. **Vehicle Count**: Multi-vehicle incidents scale severity score upwards.
4. **Vulnerable Road User Proximity**: Distance and movement classification of pedestrians near the impact zone.
5. **Persistence**: Time duration the accident scene remains disrupted.

#### Category Boundaries
- **LOW** (`score < 0.50`): Minor bump or low-speed collision without pedestrian involvement.
- **MEDIUM** (`0.50 <= score < 0.72`): Significant speed reduction, multiple vehicles, or pedestrian in medium proximity.
- **HIGH** (`score >= 0.72`): Severe impact, major speed drops, or person detected at high risk near the collision zone. Displayed with prominent red borders and `HIGH SEVERITY — MAJOR INCIDENT SUSPECTED`.
- **UNCERTAIN**: Assigned when visual evidence is borderline, conflicting, or when a model detection lacks supporting kinematic cues. Labeled honestly as `UNCERTAIN — MANUAL REVIEW`.

> **Ethical & Scientific Clarification**: Severity represents an estimated visual category, not a medical diagnosis. NexGuard does not detect blood or internal injuries.

---

## Installation & Setup

### Prerequisites
- Linux, macOS, or Windows
- Python 3.10+ (tested through Python 3.14)
- Webcam (for Option 1) or sample video files (for Option 2)

### 1. Clone the Repository
```bash
git clone https://github.com/reddy895/NexGuard.git
cd NexGuard
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
NexGuard uses standard YOLOv8 weights (e.g., `yolov8n.pt` for nano, or custom accident-trained weights). If `yolov8n.pt` is not in the directory, Ultralytics will automatically fetch it on first execution:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## Dataset, Training & MLOps Pipeline

NexGuard incorporates a scientifically rigorous dataset validation, preprocessing, model training, and evaluation pipeline.

### 1. Dataset Integrity & Validation
Validate the CCTV accident dataset to detect corrupted images, zero-byte files, and invalid formats:
```bash
python scripts/validate_dataset.py --data datasets/
```

### 2. Preprocessing & Leak-Free Splitting
Preprocess raw CCTV images into `dataset_processed/` using source-clip grouping (`SourceAwareSplitter`) to eliminate cross-split data leakage:
```bash
python -c "from nexguard.dataset.preprocessor import DatasetPreprocessor; DatasetPreprocessor('datasets').run()"
```

### 3. Model Training
Train the accident detection model on the prepared dataset:
```bash
python scripts/train_object_detector.py --config configs/accident_dataset.yaml --epochs 5 --batch 16 --device auto
```

Key training features:
- **Hardware-Aware Model Selection**: Automatically selects Nano backbones for constrained environments (<=4GB VRAM / CPU) or Small backbones for high-VRAM GPUs.
- **Safe Checkpointing**: Always saves `best.pt` and `last.pt` to `models/object_detection/`, with automated timestamped backups of prior weights.
- **Early Stopping & Overfitting Monitoring**: Tracks validation top-1 accuracy and halts when patience is exhausted.
- **Resume Support**: Pass `--resume` to resume interrupted training from `last.pt`.

### 4. Comprehensive Evaluation
Run empirical evaluation across object detection, frame-level accident classification, temporal event reasoning, and severity baseline:
```bash
python scripts/evaluate_models.py
```

Generated Reports:
- `reports/dataset_audit.md` — Comprehensive raw dataset audit and anomaly report.
- `reports/class_distribution.png` — High-resolution class and split distribution graphic.
- `reports/data_split_report.md` — Source-aware video grouping split report.
- `reports/object_detection_results.md` — Detection mAP, precision, recall, and FPS benchmarks.
- `reports/confusion_matrix.png` — Publication-ready test split confusion matrix.
- `reports/accident_detection_results.md` — Temporal multi-frame event evaluation.
- `reports/severity_results.md` — Severity assessment benchmark and safety metrics.
- `reports/final_model_evaluation.md` — Unified executive summary of model evaluation.

---

## Running the System

Start the interactive terminal application:
```bash
python main.py
```

Choose from the interactive menu:
- Press `1` and press `Enter` to open the webcam.
- Press `2` and press `Enter` to select a video file via the native file chooser.
- Press `3` and press `Enter` to exit.

To stop playback during video or webcam detection, press `q`, `Q`, or `ESC` in the display window.

---

## Testing

NexGuard includes a comprehensive test suite (55+ unit & integration tests) validating state transitions, cooldown windows, geometric calculations, dataset parsing, and error handling.

Run all tests:
```bash
pytest -v
```

---

## What Works Now vs Future Requirements

| Capability | Status | Implementation Details |
| :--- | :--- | :--- |
| **Terminal Menu Interface** | Working | Clean interactive CLI with webcam, file picker, and exit options |
| **Object Detection (COCO)** | Working | YOLOv8 inference for cars, motorcycles, buses, trucks, people |
| **Object Tracking** | Working | Centroid & IoU tracker with smoothed velocity and heading estimation |
| **Temporal Accident Confirmation** | Working | Multi-frame approach velocity, deceleration, and proximity heuristics |
| **Accident Classification** | Working | YOLOv8 Nano classifier trained on deduplicated CCTV footage (95.1% test acc) |
| **Rule-Based Severity Baseline** | Working | Continuous scoring mapped to LOW, MEDIUM, HIGH, and UNCERTAIN |
| **Incident Archiving & Alerts** | Working | Forensic snapshot capture and duplicate rate-limited alerting |
| **Medical / Injury Assessment** | Unsupported | General object detectors cannot diagnose injuries or detect blood |

---

## Known Limitations & Troubleshooting

1. **Camera Permissions**: On Linux systems, verify user permissions to access video devices (`sudo usermod -a -G video $USER`).
2. **Headless Environments**: When running in a headless server without a display (`$DISPLAY` not set), OpenCV `imshow` requires an X11 forwarder or virtual framebuffer (e.g., `xvfb-run python main.py`).
3. **Hardware Acceleration**: If an NVIDIA GPU is present, PyTorch will automatically utilize CUDA; otherwise, it falls back seamlessly to CPU inference.
4. **Single-Camera Geometry**: Centroid distances and velocities are measured in image-space pixels per frame. Camera perspective distortion and occlusions can impact proximity measurements.

---

## License

This project is licensed under the Apache 2.0 License.
