# NexGuard System Architecture Audit

## Overview
NexGuard is a terminal-first AI CCTV accident detection system that combines YOLO object detection, object tracking, motion analysis, collision candidate detection, temporal confirmation state machine, severity assessment, evidence capture, and optional WhatsApp alert integration.

## Key Components

1. **Config & Models**:
   - `config.py`: Global configuration parameters for thresholds, sizes, skipping, models.
   - `models/base/yolov8n.pt`: Default pretrained YOLO object detection model.
   - `models/custom/accident.pt`: Optional trained custom accident YOLO model.

2. **Detection & Tracking Layer (`src/`)**:
   - `src/detector.py`: YOLO model wrapper supporting device detection (CUDA/CPU) and confidence thresholds.
   - `src/tracker.py`: Persistent object tracker maintaining velocity, acceleration, and position history per track ID.
   - `src/motion_analyzer.py`: Analyzes displacement, speed, direction, deceleration, and direction change.
   - `src/collision_analyzer.py`: Analyzes bounding box overlap (IoU), proximity distance, relative approach speed, and identifies collision candidates.
   - `src/accident_detector.py`: Multi-frame temporal state machine (`NORMAL` -> `SUSPECTED_COLLISION` -> `CONFIRMING` -> `ACCIDENT_CONFIRMED` -> `RECOVERY`) with false positive suppression.
   - `src/severity_engine.py`: Maps accident indicators into severity levels (`NORMAL`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
   - `src/incident_manager.py` & `src/evidence_manager.py`: Captures snapshot evidence (`incidents/YYYY-MM-DD_HH-MM-SS.jpg`) and manages alert cooldown.
   - `src/performance.py`: Real FPS monitoring and configurable frame skip presets (Accuracy, Balanced, Performance).
   - `src/pipeline.py`: Unified pipeline orchestrating video frame processing, detection, tracking, analysis, display, and alerting.

3. **Utilities (`utils/`)**:
   - `utils/logger.py`: Structured logging to console and file.
   - `utils/geometry.py`: Spatial distance, IoU calculation, and vector geometry helper functions.
   - `utils/video.py`: Media source (Image, Video, Webcam index, RTSP URL) validation and decoding.
   - `utils/validation.py`: Input file/stream and dataset validation routines.

4. **Training & Validation (`training/`)**:
   - `training/train_accident.py`: Dataset verifier and fine-tuning wrapper.
   - `training/validate_accident.py`: Model performance evaluation script.

5. **WhatsApp Integration (`whatsapp/`)**:
   - Local Node.js / Baileys / wwebjs wrapper for session authentication and message transmission.
