# NexGuard — AI CCTV Accident Detection System (Max 15 FPS)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![FPS](https://img.shields.io/badge/FPS-Capped%2015.0%20MAX-brightgreen)
![WhatsApp](https://img.shields.io/badge/WhatsApp-Bot%20Integration-success)

NexGuard is a high-performance Edge AI CCTV accident detection system that enforces a strict **Max 15 FPS** processing cap for optimal hardware efficiency while running a background **WhatsApp Bot** to listen for commands and dispatch real-time incident alerts.

---

## 📁 Architecture & File Structure

```
NexGuard/
├── test_clips/                     # Sample test video clips
├── whatsapp_bot/                   # WhatsApp Node.js listener daemon & python client
│   ├── index.js                    # whatsapp-web.js daemon & HTTP API server
│   └── bot_client.py               # Python wrapper for status & message dispatch
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive documentation
├── config.py                       # System configuration & Max 15 FPS parameter
├── gesture_classifier.joblib       # Trained scikit-learn motion dynamics model
├── gesture_classifier_fast.npz     # NumPy binary weights for fast inference
├── main.py                         # Application CLI & live surveillance pipeline
├── requirements.txt                # Python dependencies
├── send_test_message.py            # Standalone CLI tool to dispatch test WhatsApp alerts
├── test_system.py                  # Automated test suite (15 FPS, tracker, ML model)
├── tracker.py                      # Multi-object tracker & trajectory vectors
├── train_gesture_model.py          # Synthetic dataset generator & model trainer
├── ui.py                           # Surveillance HUD dashboard renderer
└── utils.py                        # FPSLimiter rate regulator & spatial geometry math
```

---

## ✨ Features

- **Strict Max 15 FPS Cap**: Enforces hardware timing so GPU/CPU frame rate never exceeds 15.0 FPS.
- **WhatsApp Listener & Dispatcher**: Runs a background daemon using `whatsapp-web.js` listening for `!status`, `!ping`, `!help` commands while allowing instant alert dispatching from Python.
- **YOLOv8 + Motion ML Classifier**: Combines spatial YOLO object detection with a Random Forest dynamic gesture classifier (`gesture_classifier.joblib`).
- **Multi-Object Tracking**: Centroid and IoU object tracker with historical speed drop analysis.
- **Surveillance HUD Overlay**: Displays live FPS badge (Max 15 FPS), vehicle/pedestrian vectors, WhatsApp connection state, and flashing accident alert banners.

---

## 🚀 Quick Start

### 1. Requirements & Setup
```bash
pip install -r requirements.txt
cd whatsapp_bot && npm install && cd ..
```

### 2. WhatsApp QR Authentication
Scan your QR code to connect WhatsApp:
```bash
node whatsapp_bot/index.js qr
```

### 3. WhatsApp Bot Interactive Commands
When connected, you can interact with the bot from WhatsApp:
- `!ping` - Test bot responsiveness
- `!status` - Retrieve CCTV surveillance status and FPS health
- `!help` - Display available commands

### 4. Dispatch Test Message
```bash
python3 send_test_message.py --number 919876543210
```

### 5. Run System & Live Surveillance
```bash
python3 main.py
```

### 6. Run Diagnostic Test Suite
```bash
python3 test_system.py
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MAX_FPS` | `15` | Maximum FPS rate cap (enforced by `FPSLimiter`) |
| `WHATSAPP_RECIPIENT` | `""` | Target phone number with country code |
| `WHATSAPP_PORT` | `3001` | Local IPC HTTP server port |
| `NEXGUARD_DEVICE` | `cpu` | Processing device (`cpu` or `cuda`) |
| `YOLO_CONF` | `0.35` | Confidence detection threshold |

---

## 📄 License
MIT License.