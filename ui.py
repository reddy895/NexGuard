"""
NexGuard Dashboard UI & HUD Rendering Engine
Draws futuristic dark-themed surveillance overlays, live tracking vectors,
max 15 FPS performance badges, WhatsApp connectivity indicators, and accident alerts.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional
from config import config
from tracker import TrackedObject


class NexGuardUI:
    """Surveillance HUD and telemetry renderer."""
    
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.alert_flash_state = False
        self.last_flash_time = time.time()

    def draw_hud(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        is_accident: bool,
        confidence: float,
        severity: str,
        fps: float,
        whatsapp_status: str = "CONNECTED",
        incident_id: Optional[str] = None
    ) -> np.ndarray:
        """Renders complete CCTV HUD overlay on top of frame."""
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Top Control Bar (Dark Glassmorphism banner)
        cv2.rectangle(canvas, (0, 0), (w, 50), (15, 18, 24), -1)
        cv2.line(canvas, (0, 50), (w, 50), (0, 255, 200), 2)

        # Title
        cv2.putText(canvas, "NEXGUARD AI SURVEILLANCE", (20, 32), self.font, 0.7, (255, 255, 255), 2)

        # MAX 15 FPS badge
        fps_color = (0, 255, 120) if fps <= 15.1 else (0, 165, 255)
        cv2.rectangle(canvas, (370, 10), (580, 40), (30, 35, 45), -1)
        cv2.rectangle(canvas, (370, 10), (580, 40), fps_color, 1)
        cv2.putText(canvas, f"FPS: {fps:.1f} / 15.0 MAX", (380, 31), self.font, 0.55, fps_color, 2)

        # WhatsApp Status Badge
        wa_color = (0, 255, 120) if whatsapp_status.upper() == "CONNECTED" else (0, 100, 255)
        cv2.rectangle(canvas, (600, 10), (810, 40), (30, 35, 45), -1)
        cv2.rectangle(canvas, (600, 10), (810, 40), wa_color, 1)
        cv2.putText(canvas, f"WA: {whatsapp_status.upper()}", (610, 31), self.font, 0.55, wa_color, 2)

        # 2. Draw Bounding Boxes and Speed Vectors for Tracked Objects
        vehicles_count = 0
        people_count = 0

        for obj in tracked_objects:
            if obj.class_name in config.vehicle_classes:
                vehicles_count += 1
                box_color = (255, 180, 0)  # Cyan/Blue
            else:
                people_count += 1
                box_color = (180, 255, 0)  # Bright Green

            x1, y1, x2, y2 = map(int, obj.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            # Draw track ID label
            label = f"ID-{obj.track_id} {obj.class_name} ({obj.speed:.1f}px/f)"
            cv2.putText(canvas, label, (x1, max(15, y1 - 8)), self.font, 0.45, box_color, 1)

            # Motion direction vector arrow
            if len(obj.history) >= 2:
                p1 = (int(obj.history[-2][0]), int(obj.history[-2][1]))
                p2 = (int(obj.cx), int(obj.cy))
                cv2.arrowedLine(canvas, p1, p2, (0, 255, 255), 2, tipLength=0.3)

        # 3. Bottom Telemetry Bar
        cv2.rectangle(canvas, (0, h - 40), (w, h), (15, 18, 24), -1)
        cv2.line(canvas, (0, h - 40), (w, h - 40), (100, 100, 100), 1)
        telemetry_text = f"TRACKED VEHICLES: {vehicles_count} | PEDESTRIANS: {people_count} | MODEL: YOLOv8 + GESTURE ML"
        cv2.putText(canvas, telemetry_text, (20, h - 14), self.font, 0.5, (200, 200, 200), 1)

        # 4. Flashing Accident Alert Banner if Incident Triggered
        if is_accident:
            now = time.time()
            if now - self.last_flash_time > 0.4:
                self.alert_flash_state = not self.alert_flash_state
                self.last_flash_time = now

            banner_color = (0, 0, 255) if self.alert_flash_state else (0, 0, 180)
            cv2.rectangle(canvas, (w // 2 - 280, 60), (w // 2 + 280, 120), banner_color, -1)
            cv2.rectangle(canvas, (w // 2 - 280, 60), (w // 2 + 280, 120), (255, 255, 255), 2)

            alert_msg = f"ACCIDENT DETECTED! [{severity.upper()}] ({int(confidence * 100)}%)"
            cv2.putText(canvas, alert_msg, (w // 2 - 250, 95), self.font, 0.75, (255, 255, 255), 2)

            if incident_id:
                cv2.putText(canvas, f"INCIDENT ID: {incident_id}", (w // 2 - 250, 112), self.font, 0.4, (200, 255, 200), 1)

        return canvas
