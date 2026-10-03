# NexGuard Detection Pipeline Architecture

```
                       Input Stream (Image / Video / Webcam / RTSP)
                                           │
                                           ▼
                                 Media Source Validation
                                           │
                                           ▼
                              Frame Preprocessing & Skip
                                           │
                                           ▼
                                 YOLO Object Detection
                             (Base Model + Optional Custom)
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

## Two-Layer Accident Detection

### Layer 1: Rule-Based Temporal Motion & Collision Engine
- **Independent**: Works out of the box using any YOLO vehicle detection (car, truck, bus, motorcycle, bicycle).
- **Features**: Analyzes relative velocity, deceleration, overlap (IoU), and trajectory deviation over time.
- **State Machine**: Requires evidence across consecutive frames before declaring an accident to eliminate single-frame false positives.

### Layer 2: Custom Accident Model Fusion
- **Optional**: Loads `models/custom/accident.pt` if present.
- **Evidence Fusion**: Combines bounding-box classification confidence from the custom model with Layer 1 temporal evidence to strengthen detection certainty.
- **Graceful Fallback**: If `models/custom/accident.pt` is missing, system falls back to Layer 1 seamlessly.
