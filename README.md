# NEXGUARD — EDGE AI SURVEILLANCE SYSTEM

![NexGuard Banner](https://img.shields.io/badge/NexGuard-Edge_AI_Surveillance-blue?style=for-the-badge)
![Phase 1](https://img.shields.io/badge/Phase_1-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO-v8_Engine-red?style=for-the-badge)

NexGuard is an **Edge AI CCTV Surveillance & Intelligent Incident Response Platform** engineered to provide real-time automated monitoring for urban road networks and surveillance cameras.

> Default model limitation: the standard YOLOv8 COCO model detects objects such as people and vehicles but does not directly classify road accidents. NexGuard therefore combines YOLOv8 detection and tracking with temporal accident-event analysis. This is a heuristic collision detector based on multi-frame motion, proximity, and overlap signals rather than a direct accident-class prediction. A future version can use a custom accident-trained YOLOv8 model and/or a dedicated temporal action-recognition model trained on accident datasets.

---

## 📌 Overview & Problem Statement

Urban traffic monitoring traditionally relies on manual CCTV observation by human operators, leading to delayed response times during high-severity road accidents. NexGuard solves this by deploying real-time YOLO object detection directly on local edge hardware, enabling instant detection, telemetry monitoring, and automated event reporting without heavy cloud dependence.

---

## 🎯 Current Phase Scope (Phase 1)

This repository contains **Phase 1** of NexGuard — a terminal-based Edge AI surveillance prototype.

### Core Features (Phase 1)
- 🖥️ **Clean Terminal Interface**: Menu-driven interface for selecting webcam, video file, or image input.
- ⚡ **YOLO Detection Engine**: Real-time object detection with dynamic class mapping (COCO object classes: person, car, motorcycle, bus, truck, etc.).
- 🚀 **Hardware Acceleration**: Auto-detection of CUDA/GPU or host CPU runtime environment.
- 📊 **Live Telemetry Overlay**: Real-time FPS monitoring, frame count, object density tracking, and device mode display.
- 🎮 **Interactive Keyboard Controls**: Pause/resume stream, save evidence snapshots, and reset telemetry statistics.
- 📸 **Snapshot Saving**: Saves annotated evidence snapshots directly to `outputs/` directory.
- 📝 **Structured Telemetry Logging**: Application logs (`logs/nexguard.log`) and event logs (`logs/detection_events.jsonl`).

---

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **AI/ML Engine**: Ultralytics YOLOv8 / PyTorch
- **Computer Vision**: OpenCV (Open Source Computer Vision Library)
- **Configuration**: PyYAML / Dataclass configuration system
- **Testing**: Python standard `unittest` test suite

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/reddy895/NexGuard.git
cd NexGuard
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the application entry point:

```bash
python main.py
```

### Interactive Terminal Menu
```text
==================================================
                 NEXGUARD
        AI CCTV ACCIDENT DETECTION
==================================================
Model: YOLOv8n
Tracking: ByteTrack
Accident Detection: ENABLED
Waiting for video input...
==================================================

Select input:
1. Webcam
2. Video file
3. Image file
4. Exit

Enter option (1-4):
```

### Input Selection Options

1. **Webcam Stream**:  
   Select option `1`. NexGuard initializes the primary camera device (`index 0`).

2. **Video File Inference**:  
   Select option `2`. Enter path to video file or press `Enter` to run sample video:
   ```text
   Enter video file path [Press Enter for default: 'assets/sample/surveillance_sample.mp4']:
   ```

3. **Static Image Inference**:  
   Select option `3`. Enter path to image file or press `Enter` to run sample traffic image:
   ```text
   Enter image file path [Press Enter for default: 'assets/sample/traffic_sample.jpg']:
   ```

---

## 🎮 Keyboard Controls

While the live surveillance window is active, use the following interactive keys:

| Key | Action | Description |
|---|---|---|
| **`Q`** / **`ESC`** | **Quit** | Stop detection loop and return to menu |
| **`P`** | **Pause / Resume** | Freeze stream for detailed frame inspection |
| **`S`** | **Snapshot** | Save annotated frame to `outputs/detection_YYYYMMDD_HHMMSS.jpg` |
| **`R`** | **Reset Stats** | Reset frame counter, FPS timer, and object tally |

---

## ⚙️ Configuration System

Configure settings via `config.py` or `config.yaml`:

```yaml
model_path: yolov8n.pt
confidence_threshold: 0.25
iou_threshold: 0.45
device: auto           # Options: 'auto', 'cuda', 'cpu'
camera_index: 0
input_size: 640
frame_skip: 0
output_dir: outputs
log_dir: logs
refresh_rate_sec: 1.0
```

---

## 📂 Project Structure

```text
NexGuard/
│
├── main.py                     # Main application entry point & CLI menu
├── config.py                   # Configuration management system
├── config.yaml                 # Configuration YAML file
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
├── .gitignore                  # Git ignore rules for AI & cache files
│
├── models/                     # Directory for YOLO model weights (.pt)
│
├── src/                        # Application source code
│   ├── __init__.py
│   ├── detector.py             # YOLO detector engine abstraction
│   ├── video.py                # Video & image input sources (Webcam, File, Image)
│   ├── display.py              # Visualizations & telemetry overlay renderer
│   ├── logger.py               # Application logger
│   ├── event_logger.py         # Structured event telemetry logger
│   └── utils.py                # Hardware detection & utility functions
│
├── tests/                      # Comprehensive test suite
│   ├── __init__.py
│   ├── test_config.py          # Configuration unit tests
│   ├── test_utils.py           # Utility function tests
│   ├── test_display.py         # Display renderer unit tests
│   ├── test_detector.py        # YOLO detector module tests
│   ├── test_video.py           # Video input module tests
│   ├── test_event_logger.py    # Event logger unit tests
│   ├── test_main.py            # Main entry point unit tests
│   └── test_integration.py     # End-to-end pipeline integration tests
│
├── assets/                     # Sample assets & generator
│   └── sample/
│       ├── generate_samples.py # Synthetic sample generator
│       ├── traffic_sample.jpg  # Generated sample traffic image
│       └── surveillance_sample.mp4 # Generated sample surveillance video
│
├── logs/                       # Application & event logs (gitignored)
└── outputs/                    # Evidence snapshot output directory (gitignored)
```

---

## 🧪 Testing & Verification

Run the full test suite via standard `unittest`:

```bash
python3 -m unittest discover tests
```

---

## 🛣️ Future Architecture Roadmap

NexGuard is designed modularly to support multi-phase expansion:

- **Phase 1 (Completed)**: Basic Edge AI CCTV surveillance & terminal YOLO pipeline.
- **Phase 2**: Real-time road accident detection algorithms.
- **Phase 3**: Temporal multi-frame motion verification & collision velocity tracking.
- **Phase 4**: Incident severity classification (Minor, Moderate, Critical).
- **Phase 5**: GPS location intelligence & GIS coordinate mapping.
- **Phase 6**: Smart routing for nearest police & hospital emergency dispatch.
- **Phase 7**: Automated WhatsApp, SMS, and emergency broadcast alerts.
- **Phase 8**: Live web command dashboard & multi-camera streaming feed.
- **Phase 9**: Predictive accident hotspot analytics & historical heatmaps.
- **Phase 10**: Multi-camera edge node correlation & spatial tracking network.

---

*NexGuard Edge AI Surveillance System — Phase 1 Release*
