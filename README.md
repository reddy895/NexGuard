# NexGuard — AI-Powered CCTV Accident Detection System

NexGuard is a modular Python computer-vision system designed for edge surveillance and real-time traffic accident response. It processes live CCTV feeds (RTSP/Webcam) and video files to detect, track, and analyze potential traffic accidents.

Unlike simple frame-by-frame classifiers, NexGuard uses temporal event analysis to evaluate object trajectories, collision indicators, and severity before generating incident alerts.

## 🏗️ Architecture

The following Mermaid diagram represents the complete NexGuard pipeline, including video input, YOLOv8 detection, object tracking, temporal collision analysis, severity scoring, incident management, evidence recording, and WhatsApp notifications.

```mermaid
flowchart TD
    %% Input Source Layer
    subgraph INPUT["1. Input Source Layer - nexguard/input"]
        direction LR
        SRC1["Webcam 0"]
        SRC2["Video File (.mp4)"]
        SRC3["RTSP IP Camera"]
    end

    MAIN["Pipeline Orchestrator<br/>main.py"]

    INPUT -->|Frame Feed| MAIN

    %% Core Computer Vision Pipeline
    subgraph CV["2. Core CV and Heuristic Engine"]
        direction TB

        DET["YOLOv8 Detection Engine<br/>nexguard/detection"]

        TRACK["Temporal Object Tracker<br/>IoU and Velocity Estimation<br/>nexguard/tracking"]

        COLL["Collision Heuristics<br/>Temporal Window Analysis<br/>nexguard/accident"]

        SEV["Severity Scoring<br/>LOW / MEDIUM / HIGH / CRITICAL<br/>nexguard/accident/severity.py"]

        LIFECYCLE["Incident Lifecycle Engine<br/>nexguard/accident/event.py"]

        DET -->|Bounding Boxes and Classes| TRACK
        TRACK -->|Persistent IDs and Trajectories| COLL
        COLL -->|Collision Candidates| SEV
        SEV -->|Scored Event Details| LIFECYCLE
    end

    MAIN -->|Raw Frames| DET

    %% Visualization and Diagnostics
    subgraph VIS["Visualization and Diagnostics"]
        direction LR
        HUD["OpenCV HUD Window<br/>nexguard/visualization"]
        TERM["Terminal Statistics and Logs<br/>nexguard/utils"]
    end

    MAIN -.->|Render Output| HUD
    MAIN -.->|CLI Output| TERM

    %% Asynchronous Processing
    subgraph ASYNC["3. Non-Blocking Background Tasks"]
        direction TB

        QUEUE[("Async I/O Queue")]

        EVID["Evidence Recorder<br/>nexguard/evidence"]

        STORAGE[("Local Evidence Storage<br/>.mp4 and .jpg")]

        ALERT["Alert Client<br/>nexguard/alerts"]

        BOT["WhatsApp Microservice<br/>whatsapp_bot / whatsapp-web.js"]

        CONTACTS[("Emergency Contacts<br/>WhatsApp")]

        QUEUE -->|Background Worker| EVID
        QUEUE -->|Background Worker| ALERT

        EVID -->|Save Impact Frame and Video Clip| STORAGE

        ALERT -->|IPC or HTTP Trigger| BOT

        BOT -->|Send Incident Alert| CONTACTS
    end

    LIFECYCLE -->|Confirmed Incident Event| QUEUE

    %% Styling
    classDef input fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef pipeline fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef async fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef visualization fill:#ede7f6,stroke:#512da8,stroke-width:2px,color:#311b92;
    classDef core fill:#eceff1,stroke:#455a64,stroke-width:3px,color:#263238;

    class SRC1,SRC2,SRC3 input;
    class DET,TRACK,COLL,SEV,LIFECYCLE pipeline;
    class QUEUE,EVID,STORAGE,ALERT,BOT,CONTACTS async;
    class HUD,TERM visualization;
    class MAIN core;
```

### Architecture Workflow

1. **Input:** Capture frames from a webcam, video file, or RTSP camera.
2. **Detection:** YOLOv8 identifies supported objects in each frame.
3. **Tracking:** Associate detections across frames and estimate object trajectories.
4. **Collision analysis:** Evaluate temporal motion and collision indicators.
5. **Severity assessment:** Score candidate incidents using configured heuristics.
6. **Incident management:** Track incident states and confirm events.
7. **Evidence collection:** Save the selected impact frame and surrounding video footage.
8. **Alert delivery:** Route incident information to the WhatsApp microservice.
9. **Monitoring:** Display annotated frames and diagnostic information when enabled.

## ✨ Key Features

- **Temporal Object Tracking:** IoU-based association and velocity estimation for tracking objects across frames.
- **Accident Detection:** Evaluates collision indicators across a temporal window instead of relying solely on individual frames.
- **Severity Classification:** Supports LOW, MEDIUM, HIGH, and CRITICAL incident levels based on configured collision and scene-related indicators.
- **Automated Evidence Collection:** Saves impact frames and surrounding video clips to the local `evidence/` directory.
- **WhatsApp Notifications:** Uses a Node.js `whatsapp-web.js` microservice with QR-code authentication.
- **Non-Blocking Processing:** Background workers handle evidence and notification tasks to reduce interruptions to the inference loop.
- **OpenCV Visualization:** Provides an annotated video window and real-time diagnostics.
- **Modular Architecture:** Separates detection, tracking, accident analysis, alerts, evidence, and video input into independent modules.

## 🚀 Installation

### 1. Prerequisites

- Python 3.8 or later, compatible with the project's dependencies.
- Node.js and npm compatible with the WhatsApp bot dependencies.
- FFmpeg and the required OpenCV system libraries.
- A webcam, RTSP camera, or test video.
- A compatible YOLO model.

On Ubuntu/Debian, install the common system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6
```

Install Chromium if required by your WhatsApp bot configuration and your distribution provides it:

```bash
sudo apt-get install -y chromium
```

### 2. Clone the Repository

```bash
git clone https://github.com/reddy895/NexGuard.git
cd NexGuard
```

### 3. Create a Python Environment

```bash
python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the WhatsApp Bot

```bash
cd whatsapp_bot
npm install
cd ..
```

Use `npm install --ignore-scripts` only if the project's setup instructions require it. Some packages need installation scripts to download or configure runtime dependencies.

### 5. Configure Environment Variables

Create a local environment file from the example:

```bash
cp .env.example .env
```

Edit `.env` and configure the appropriate settings, including:

- YOLO model path.
- Default video source.
- WhatsApp enable/disable setting.
- Emergency contact number.
- Accident confidence and severity thresholds, if supported by your configuration.

Use the variable names defined in your actual `.env.example` and `config.py`. Do not commit `.env`, WhatsApp session credentials, or authentication data to GitHub.

## 💻 Usage

Activate your virtual environment before running NexGuard.

### Run With the Default Webcam

```bash
python main.py
```

### Run With a Video File

```bash
python main.py --source test_clips/road.mp4
```

### Run With an RTSP Camera

```bash
python main.py --source "rtsp://username:password@CAMERA_IP:554/stream"
```

Replace the credentials, camera address, and stream path with your actual camera settings.

### Run Without the OpenCV Display

```bash
python main.py --no-display
```

### Display Available Options

```bash
python main.py --help
```

### Test WhatsApp Integration

If the repository contains `send_test_message.py`, run:

```bash
python send_test_message.py --target YOUR_TEST_NUMBER
```

Use the phone-number format expected by the script. Verify the WhatsApp session and recipient configuration before enabling automated alerts.

## 🛠️ Project Structure

```text
NexGuard/
├── main.py
├── config.py
├── requirements.txt
├── .env.example
│
├── nexguard/
│   ├── accident/
│   │   ├── severity.py
│   │   └── event.py
│   ├── alerts/
│   ├── detection/
│   ├── evidence/
│   ├── input/
│   ├── models/
│   ├── tracking/
│   ├── utils/
│   └── visualization/
│
├── whatsapp_bot/
├── tests/
├── scripts/
│   └── train_accident_model.py
├── datasets/
│   └── README.md
├── test_clips/
└── README.md
```

*This tree illustrates the intended module organization. Actual filenames and optional files may differ in the repository.*

## 🧪 Testing

Run the local test suite:

```bash
python -m pytest tests/ -v
```

To run the tests with additional diagnostic output:

```bash
python -m pytest tests/ -vv
```

The test suite is intended to validate core functionality without requiring a live camera or WhatsApp connection. Individual tests may still require dependencies or fixtures documented by the repository.

## 🧠 Custom Accident Model Training

NexGuard uses YOLO-based object detection. A general-purpose pretrained model such as `yolov8n.pt` is not, by itself, a reliable traffic-accident classifier. Accident detection depends on the trained model, temporal heuristics, and validation against representative footage.

### 1. Prepare the Dataset

Review the dataset instructions:

```bash
cat datasets/README.md
```

Ensure the dataset is organized according to the format expected by the training script and that `datasets/accident/data.yaml` exists.

### 2. Train the Model

```bash
python scripts/train_accident_model.py \
  --data datasets/accident/data.yaml \
  --epochs 100
```

The dataset must contain suitable labeled examples and the class definitions expected by the detector and accident-analysis modules.

### 3. Configure the Trained Weights

After training, identify the actual output weights, such as `best.pt`, and update `.env`:

```dotenv
NEXGUARD_MODEL_PATH=path/to/your/best.pt
```

Replace the example path with the real path to your trained weights. Confirm that `config.py` reads this exact environment variable.

### 4. Validate the Model

Evaluate the trained model on separate validation footage, including normal traffic, near misses, different camera angles, low-light scenes, and genuine collision footage where available.

Measure false positives and missed incidents before using the system in real-world surveillance.

## 🔐 Safety and Deployment Notes

- NexGuard should be treated as an assistive detection system, not a replacement for verified emergency dispatch.
- Validate incident alerts with representative footage before enabling operational notifications.
- Use explicit severity thresholds and carefully tested routing rules.
- Protect camera credentials, contact numbers, WhatsApp sessions, and stored evidence.
- Configure evidence retention and access controls according to applicable privacy requirements.
- Do not assume an alert has been delivered merely because an event was detected; record delivery status and handle failures where supported.

## 🤝 Contributing

Contributions that improve detection accuracy, tracking stability, evidence reliability, test coverage, and alert delivery are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make focused changes and add tests.
4. Run the test suite.
5. Submit a pull request describing the change and its validation.

## 📄 License

Add the project's selected license and corresponding `LICENSE` file before distributing NexGuard. Until then, do not assume that the repository is released under a particular open-source license.

---

**NexGuard — Detect incidents. Preserve evidence. Accelerate response.**
