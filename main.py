#!/usr/bin/env python3
"""
NexGuard — Terminal-First AI CCTV Accident Detection System
Main Entry Point & Live Surveillance Pipeline Daemon.
Strictly caps processing at Max 15 FPS and manages WhatsApp alert dispatching.
"""

import os
import sys
import time
import math
import cv2
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import config
from tracker import ObjectTracker, TrackedObject
from utils import FPSLimiter, compute_bbox_iou, compute_centroid_distance, log_event
from ui import NexGuardUI
from whatsapp_bot.bot_client import whatsapp_bot


class NexGuardPipeline:
    """Core AI CCTV Processing Pipeline (Max 15 FPS + WhatsApp Dispatcher)."""
    
    def __init__(self):
        self.tracker = ObjectTracker(max_lost_frames=config.max_lost_frames, iou_threshold=config.iou_threshold)
        self.fps_limiter = FPSLimiter(target_fps=config.max_fps)
        self.ui = NexGuardUI()
        
        # Load YOLO model dynamically
        self.yolo_model = None
        self._init_yolo_model()

        # Load Motion Gesture Classifier
        self.gesture_clf = None
        self.gesture_scaler = None
        self._init_gesture_classifier()

        # Temporal Confirmation State
        self.consecutive_accident_frames = 0
        self.in_accident_state = False
        self.last_incident_id = None

        # Start WhatsApp listener daemon in background
        whatsapp_bot.start_listener_daemon()

    def _init_yolo_model(self):
        try:
            from ultralytics import YOLO
            model_p = Path(config.yolo_model_path)
            if model_p.exists():
                self.yolo_model = YOLO(str(model_p))
                log_event("info", f"Loaded YOLOv8 model from {model_p}")
            else:
                self.yolo_model = YOLO("yolov8n.pt")
                log_event("info", "Initialized default YOLOv8n model.")
        except Exception as e:
            log_event("warning", f"YOLO model loading note: {e}. Falling back to dynamic computer vision detector.")
            self.yolo_model = None

    def _init_gesture_classifier(self):
        p = Path(config.gesture_model_joblib)
        if p.exists():
            try:
                data = joblib.load(p)
                self.gesture_clf = data["model"]
                self.gesture_scaler = data["scaler"]
                log_event("info", f"Loaded Motion Gesture Classifier from {p}")
            except Exception as e:
                log_event("warning", f"Gesture classifier load error: {e}")

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Runs object detection on frame using YOLO or CV fallback."""
        detections = []
        if self.yolo_model is not None:
            results = self.yolo_model(frame, verbose=False, conf=config.yolo_confidence)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names.get(cls_id, f"class_{cls_id}")
                
                # Filter vehicles and pedestrians
                if cls_name in config.vehicle_classes or cls_name in config.person_classes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append({
                        "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        "class_name": cls_name,
                        "confidence": conf
                    })
        else:
            # High-performance OpenCV motion fallback detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 800:
                    x, y, w, h = cv2.boundingRect(cnt)
                    detections.append({
                        "bbox": (float(x), float(y), float(x + w), float(y + h)),
                        "class_name": "car" if w * h > 3000 else "person",
                        "confidence": 0.80
                    })
        return detections

    def analyze_frame_dynamics(self, tracked_objects: List[TrackedObject]) -> Dict[str, Any]:
        """Evaluates collision candidate indicators and gesture classifier predictions."""
        is_accident = False
        confidence = 0.0
        severity = "NORMAL"
        involved_ids = []

        vehicles = [obj for obj in tracked_objects if obj.class_name in config.vehicle_classes]

        # Check vehicle interactions
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1 = vehicles[i]
                v2 = vehicles[j]

                iou = compute_bbox_iou(v1.bbox, v2.bbox)
                dist = compute_centroid_distance(v1.bbox, v2.bbox)
                s_drop1 = v1.get_speed_drop_ratio()
                s_drop2 = v2.get_speed_drop_ratio()

                # ML Gesture classifier prediction if loaded
                if self.gesture_clf is not None and self.gesture_scaler is not None:
                    feat = np.array([[max(s_drop1, s_drop2), 25.0, iou, dist, 0.20, 15.0]])
                    feat_scaled = self.gesture_scaler.transform(feat)
                    ml_pred = self.gesture_clf.predict(feat_scaled)[0]
                    if ml_pred == 1:
                        is_accident = True
                        confidence = max(confidence, 0.85 + iou * 0.15)
                        involved_ids.extend([v1.track_id, v2.track_id])
                
                # Rule-based dynamic fallback
                if iou >= config.collision_iou_threshold or (dist <= config.proximity_threshold_px and max(s_drop1, s_drop2) > config.speed_drop_threshold):
                    is_accident = True
                    confidence = max(confidence, 0.80)
                    involved_ids.extend([v1.track_id, v2.track_id])

        # Assign severity level based on involved entities
        if is_accident:
            num_inv = len(set(involved_ids))
            if num_inv >= 3:
                severity = "CRITICAL"
            elif num_inv == 2:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

        return {
            "is_accident": is_accident,
            "confidence": min(1.0, confidence),
            "severity": severity,
            "involved_ids": list(set(involved_ids))
        }

    def process_video(self, source: Any, source_name: str = "Live Feed"):
        """Main Video Stream Loop enforced at MAX 15 FPS."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log_event("error", f"Unable to open video source: {source}")
            return

        log_event("info", f"Starting surveillance feed '{source_name}'. Enforcing MAX 15.0 FPS.")
        self.tracker.reset()
        self.consecutive_accident_frames = 0
        wa_status_info = whatsapp_bot.check_status()

        try:
            while True:
                self.fps_limiter.start_frame()
                ret, frame = cap.read()
                if not ret or frame is None:
                    log_event("info", f"End of stream reached: {source_name}")
                    break

                # Resize to standard surveillance HUD dimensions
                frame = cv2.resize(frame, (config.display_width, config.display_height))

                # 1. Detect & Track Objects
                detections = self.detect_objects(frame)
                tracked_objects = self.tracker.update(detections)

                # 2. Analyze Dynamic Motion
                analysis = self.analyze_frame_dynamics(tracked_objects)

                # 3. Temporal Confirmation State Machine
                if analysis["is_accident"]:
                    self.consecutive_accident_frames += 1
                else:
                    self.consecutive_accident_frames = max(0, self.consecutive_accident_frames - 1)

                is_confirmed_accident = self.consecutive_accident_frames >= config.accident_confirm_frames

                # 4. Trigger WhatsApp Alert on Incident Confirmation
                if is_confirmed_accident and not self.in_accident_state:
                    self.in_accident_state = True
                    self.last_incident_id = f"INC-{int(time.time())}"
                    log_event("warning", f"ACCIDENT CONFIRMED! Generating alert ID: {self.last_incident_id}")

                    # Dispatch WhatsApp message
                    whatsapp_bot.send_accident_alert(
                        incident_id=self.last_incident_id,
                        severity_level=analysis["severity"],
                        confidence=analysis["confidence"],
                        detected_summary=f"Vehicles: {len(analysis['involved_ids'])}",
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                elif not analysis["is_accident"] and self.consecutive_accident_frames == 0:
                    self.in_accident_state = False

                # 5. Enforce Max 15 FPS Sleep
                fps = self.fps_limiter.sleep_if_needed()

                # 6. Render HUD Overlay
                annotated = self.ui.draw_hud(
                    frame=frame,
                    tracked_objects=tracked_objects,
                    is_accident=self.in_accident_state,
                    confidence=analysis["confidence"],
                    severity=analysis["severity"],
                    fps=fps,
                    whatsapp_status=wa_status_info["status"],
                    incident_id=self.last_incident_id
                )

                cv2.imshow(config.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    log_event("info", "User requested quit.")
                    break

        except KeyboardInterrupt:
            log_event("info", "Process terminated by Ctrl+C.")
        finally:
            cap.release()
            cv2.destroyAllWindows()


def print_menu():
    print("\n==================================================")
    print("      NEXGUARD AI CCTV ACCIDENT SYSTEM")
    print("==================================================")
    print("  MAX FPS LIMIT     : 15.0 FPS (STRICTLY ENFORCED)")
    print("  WHATSAPP BOT      : RUNNING & LISTENING")
    print("==================================================")
    print("1. Video File Detection")
    print("2. Live Webcam Detection")
    print("3. RTSP CCTV Feed Detection")
    print("4. Train Motion / Gesture Model")
    print("5. Send Test WhatsApp Alert Message")
    print("6. Run System Diagnostics & Tests")
    print("7. Exit")
    print("==================================================")


def main():
    pipeline = NexGuardPipeline()

    while True:
        print_menu()
        choice = input("Select option (1-7): ").strip()

        if choice == "1":
            p = input("Enter video file path: ").strip().strip('"').strip("'")
            if p and Path(p).exists():
                pipeline.process_video(p, source_name=Path(p).name)
            else:
                print("[-] Error: Video file not found.")
        elif choice == "2":
            print("[+] Accessing Webcam (Device 0)...")
            pipeline.process_video(0, source_name="Webcam Feed")
        elif choice == "3":
            rtsp_url = input("Enter CCTV RTSP URL: ").strip()
            if rtsp_url:
                pipeline.process_video(rtsp_url, source_name="RTSP CCTV")
        elif choice == "4":
            print("\n[+] Retraining Motion Gesture ML Model...")
            from train_gesture_model import train_and_save_model
            train_and_save_model()
            pipeline._init_gesture_classifier()
        elif choice == "5":
            from send_test_message import main as send_test
            send_test()
        elif choice == "6":
            print("\n[+] Running System Diagnostics & Automated Test Suite...")
            os.system("python3 test_system.py")
        elif choice == "7":
            print("\nExiting NexGuard. Goodbye!\n")
            sys.exit(0)
        else:
            print("[-] Invalid option. Enter 1-7.")


if __name__ == "__main__":
    main()
