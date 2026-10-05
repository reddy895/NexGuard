# NexGuard: AI-Powered CCTV Accident Detection System

NexGuard is a clean, modular, production-grade Python computer-vision system designed for edge-surveillance and real-time accident response. It processes live CCTV feeds (RTSP/Webcam) or video files to detect, track, and confirm traffic accidents autonomously.

Unlike generic dashboards or simple frame-by-frame classifiers, NexGuard uses a robust temporal event-analysis engine. It tracks objects across time to confirm collisions with high confidence before dispatching severity-based WhatsApp alerts to emergency contacts.

---

## 🏗 Architecture

NexGuard operates entirely via Python and the terminal, completely decoupled from complex web frontends. The core pipeline is orchestrated in `main.py` and structured as follows:

```
REAL-TIME CCTV / VIDEO
        ↓
YOLO-BASED OBJECT DETECTION (nexguard/detection)
        ↓
OBJECT TRACKING (nexguard/tracking)
        ↓
TEMPORAL COLLISION HEURISTICS (nexguard/accident)
        ↓
SEVERITY & PERSON INVOLVEMENT (nexguard/accident/severity.py)
        ↓
INCIDENT LIFECYCLE MANAGEMENT (nexguard/accident/event.py)
        ↓
EVIDENCE RECORDING (nexguard/evidence)
        ↓
WHATSAPP NOTIFICATIONS (nexguard/alerts)
```

## ✨ Key Features

- **Real-Time Temporal Tracking**: IoU-based tracking with Kalman-like velocity estimations. Decisions are made over a temporal window (e.g., 20 frames), not single frames.
- **Accident Severity Classification**: Dynamically scores accidents as LOW, MEDIUM, HIGH, or CRITICAL based on collision confidence, speed drops, number of vehicles, and proximity to pedestrians.
- **Automated Evidence Collection**: Saves the highest-confidence impact frame and a surrounding 60-frame video clip (`.mp4`) automatically to `evidence/`.
- **Zero-Setup WhatsApp Alerts**: Uses a headless Node.js bot (`whatsapp-web.js`) via QR code login. No Twilio, API keys, or Meta business accounts required. Just scan the QR in your terminal.
- **Non-Blocking Architecture**: I/O operations (like saving video clips or sending WhatsApp requests) run in background threads, ensuring the inference loop never drops frames.
- **OpenCV HUD**: Real-time terminal stats alongside a clean, interactive OpenCV visualization window.

---

## 🚀 Installation

### 1. Prerequisites
- **Python 3.8+**
- **Node.js v16+** (Required for the WhatsApp QR Bot)
- **System Chrome/Chromium** (Used by the WhatsApp bot in headless mode)

```bash
# On Ubuntu/Debian:
sudo apt-get install -y ffmpeg libsm6 libxext6 chromium-browser
```

### 2. Setup Python Environment

```bash
git clone <your-repo>/NexGuard.git
cd NexGuard

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Setup WhatsApp Bot

```bash
cd whatsapp_bot
npm install --ignore-scripts
cd ..
```

### 4. Configuration

Copy the example configuration:
```bash
cp .env.example .env
```

Edit `.env` to set your preferences. By default, it uses webcam `0` and targets phone number `9591152862` for alerts.

---

## 💻 Usage

Run the main pipeline from the terminal. If WhatsApp is enabled, a QR code will print to the terminal on the first run. **Scan it with WhatsApp on your phone (Linked Devices).**

```bash
# Run with defaults (webcam 0)
python main.py

# Run with a video file
python main.py --source test_clips/road.mp4

# Run with an IP Camera (RTSP stream)
python main.py --source rtsp://admin:password@192.168.1.100:554/stream

# Run purely in terminal (no OpenCV video window)
python main.py --no-display

# Show all options
python main.py --help
```

### Testing WhatsApp Integration

Before running the full pipeline, verify your WhatsApp connection:
```bash
python send_test_message.py --target 9591152862
```

---

## 🛠 Project Structure

```text
NexGuard/
├── main.py                  # Pipeline orchestrator
├── config.py                # Centralized configuration (reads .env)
├── nexguard/
│   ├── accident/            # Core logic: collision math, severity, incidents
│   ├── alerts/              # Routing rules and WhatsApp Python client
│   ├── detection/           # YOLOv8 wrapper
│   ├── evidence/            # Frame & video clip recording
│   ├── input/               # Multi-source video handling
│   ├── models/              # Pydantic-like dataclass schemas
│   ├── tracking/            # Multi-object tracker
│   ├── utils/               # Structured logging
│   └── visualization/       # OpenCV drawing tools
├── whatsapp_bot/            # Node.js whatsapp-web.js microservice
├── tests/                   # Extensive Pytest suite
├── scripts/                 # Training scripts (YOLO fine-tuning)
└── datasets/                # Custom training dataset docs
```

---

## 🧪 Testing

The codebase includes an extensive Pytest suite that runs completely locally (no external dependencies, cameras, or WhatsApp connections required).

```bash
python -m pytest tests/ -v
```

## 🧠 Custom Training

NexGuard is designed to be highly modular. By default, it uses `yolov8n.pt` (COCO). For production accident detection, you must fine-tune a model on an accident dataset.

See `datasets/README.md` and use the included training script:
```bash
python scripts/train_accident_model.py --data datasets/accident/data.yaml --epochs 100
```
Then update `.env` to point `NEXGUARD_MODEL_PATH` to your new weights.
