# NexGuard — AI CCTV Accident Detection System

NexGuard is a terminal-first AI CCTV accident detection system built with Python, OpenCV, YOLOv8, object tracking, temporal motion analysis, and optional WhatsApp alert integration.

---

## 🚀 Key Features

1. **Terminal-First Interactive Control**: Operates entirely via clean terminal menus and OpenCV visual HUD windows.
2. **Two-Layer Accident Detection Engine**:
   - **Layer 1 (Core)**: YOLO object detection + persistent vehicle tracking + temporal motion & collision analysis. Works out-of-the-box without requiring custom training.
   - **Layer 2 (Optional)**: Custom accident YOLO model (`models/custom/accident.pt`). When present, fuses custom bounding-box evidence with Layer 1 temporal analysis. When absent, system falls back to Layer 1 seamlessly.
3. **Red Bounding Box Visual Highlighting**: When an accident is confirmed, involved vehicle bounding boxes turn **RED** (`CAR #23 ACCIDENT`) while non-involved vehicles retain normal color coding.
4. **Temporal State Machine & False-Positive Suppression**:
   - Multi-frame verification (`NORMAL` ➔ `SUSPECTED_COLLISION` ➔ `CONFIRMING` ➔ `ACCIDENT_CONFIRMED` ➔ `RECOVERY`).
   - Suppresses false positives from parallel lane passing vehicles, normal braking, and pedestrian proximity.
5. **Incident Severity & Evidence Capture**:
   - Automatically classifies incident severity into `NORMAL`, `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`.
   - Saves evidence snapshot images and metadata JSON files to `incidents/`.
6. **Local File Path Input & Media Handling**: Direct terminal file path input for images, videos (`.mp4`, `.avi`, `.mkv`), webcam feeds, and CCTV RTSP streams.
7. **Performance Tuning & Empirical FPS Monitoring**: Real FPS calculation, configurable frame skipping (`PROCESS_EVERY_N_FRAMES`), and performance modes (`Accuracy`, `Balanced`, `Performance`).
8. **CUDA GPU / CPU Acceleration**: Automatic hardware detection (`CUDA` or `CPU`).

---

## 🛠️ System Architecture

```
                       Input Stream (Image / Video / Webcam / RTSP)
                                           │
                                           ▼
                                 Media Path Validation
                                           │
                                           ▼
                              Frame Preprocessing & Skip
                                           │
                                           ▼
                                 YOLO Object Detection
                            (yolov8n.pt + Optional Custom)
                                           │
                                           ▼
                                 Persistent Tracking
                           (Track ID, Velocity, Acceleration)
                                           │
                                           ▼
                                 Track History Buffer
                                           │
                                           ▼
                                    Motion Analysis
                       (Displacement, Speed, Direction Change)
                                           │
                                           ▼
                              Vehicle Proximity & Overlap
                                    (IoU, Distance)
                                           │
                                           ▼
                               Collision Candidate Engine
                                           │
                                           ▼
                         Temporal Confirmation State Machine
                    (NORMAL -> SUSPECTED -> CONFIRMING -> CONFIRMED)
                                           │
                                           ▼
                                Accident Severity Engine
                     (NORMAL, LOW, MEDIUM, HIGH, CRITICAL)
                                           │
                                           ▼
                                Incident & Evidence Manager
                               (Save Image / JSON, Cooldown)
                                           │
                                           ▼
                                Visualization & Alerting
                 (Red BBoxes for Involved Vehicles, HUD Overlay, WhatsApp)
```

---

## 💻 Quick Start Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/NexGuard.git
cd NexGuard

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch NexGuard
```bash
python main.py
```

---

## 📂 Project Structure

```
NexGuard/
├── main.py                   # Main terminal CLI entry point
├── config.py                 # Centralized configuration system
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # Open-source license
├── .env.example              # Environment variables template
│
├── models/
│   ├── base/
│   │   └── yolov8n.pt        # Base pretrained YOLO model
│   └── custom/
│       └── accident.pt       # Optional custom accident YOLO model
│
├── src/
│   ├── detector.py           # YOLO inference wrapper & confidence filter
│   ├── tracker.py            # Persistent vehicle tracker & history buffer
│   ├── accident_detector.py  # Temporal state machine & evidence fusion
│   ├── motion_analyzer.py    # Velocity, displacement, deceleration analysis
│   ├── collision_analyzer.py # Proximity, IoU, collision candidate detection
│   ├── severity_engine.py    # Incident severity classification engine
│   ├── incident_manager.py   # Incident records & alert cooldown control
│   ├── evidence_manager.py   # Local image & JSON evidence persistence
│   ├── performance.py        # Real FPS counter & frame skip controller
│   ├── display.py            # HUD renderer & involved vehicle RED highlighting
│   └── pipeline.py           # Core application execution pipeline
│
├── utils/
│   ├── logger.py             # Structured logging system
│   ├── geometry.py           # Spatial distance, IoU, and vector math
│   ├── video.py              # Stream decoding & frame resizing
│   └── validation.py         # Path, stream, and dataset validators
│
├── training/
│   ├── train_accident.py     # Custom accident model training pipeline
│   ├── validate_accident.py  # Model validation script
│   └── dataset/              # Training dataset directory
│
├── whatsapp/                 # WhatsApp integration wrapper
├── incidents/                # Saved evidence snapshots & JSON metadata
├── outputs/                  # Exported video outputs
├── logs/                     # System logs
└── tests/                    # Comprehensive unit tests
```

---

## 🧪 Dataset Preparation & Training

To fine-tune a custom accident YOLO model, place your labeled dataset in `training/dataset/` using YOLO format:

```
training/dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Run training from the menu (Option 5) or command line:
```bash
python training/train_accident.py
```

---

## 🧪 Running Unit Tests

Run the full test suite using `pytest`:
```bash
pytest tests/
```

---

## ⚠️ Disclaimer

NexGuard is an engineering detection system designed for automated surveillance assistance and risk monitoring. It should not be used as a certified emergency medical response or critical safety system.