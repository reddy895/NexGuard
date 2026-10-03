"""
NexGuard Professional OpenCV Display Renderer (1280x720 Window Support)
"""

import time
from typing import List, Dict, Any
import cv2
import numpy as np

from utils.config import settings
from utils.detector import DetectionResult, CLASS_COLORS, DEFAULT_COLOR
from utils.tracker import TrackedObject
from utils.accident_analyzer import AccidentAnalysisResult


class CVDisplayRenderer:
    def __init__(self, window_name: str = "NexGuard - Live Detection"):
        self.window_name = window_name
        self.window_initialized = False

    def init_window(self):
        if not self.window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT)
            self.window_initialized = True

    def letterbox_resize(
        self,
        frame: np.ndarray,
        target_w: int = settings.DISPLAY_WIDTH,
        target_h: int = settings.DISPLAY_HEIGHT
    ) -> np.ndarray:
        """Resizes frame maintaining aspect ratio with black letterboxing background."""
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        dx = (target_w - nw) // 2
        dy = (target_h - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        tracked_objects: List[TrackedObject],
        analysis: AccidentAnalysisResult,
        fps: float,
        model_name: str = "ACCIDENT-YOLO",
        whatsapp_status: str = "NOT CONNECTED",
        incident_id: str = None
    ) -> np.ndarray:
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Bounding Boxes & Track IDs
        for trk in tracked_objects:
            color = CLASS_COLORS.get(trk.class_name, DEFAULT_COLOR)
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{trk.class_name.upper()} #{trk.track_id} {trk.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, max(th + 6, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(th + 2, y1 - 4)),
                font,
                font_scale,
                (10, 15, 15),
                thickness,
                lineType=cv2.LINE_AA
            )

        # Draw any custom 'accident' class detections
        for det in detections:
            if det.class_name == "accident":
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                lbl = f"ACCIDENT {det.confidence:.2f}"
                cv2.putText(annotated, lbl, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. Top-Left Badge: NEXGUARD LIVE
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(annotated, (15, 15), (200, 48), (7, 16, 17), -1)
        cv2.rectangle(annotated, (15, 15), (200, 48), (27, 50, 52), 1)
        cv2.putText(annotated, "NEXGUARD  LIVE", (25, 38), font, 0.55, (214, 255, 203), 2, cv2.LINE_AA)

        # 3. Top-Right Badge: FPS & MODEL
        info_str = f"FPS: {int(fps)} | MODEL: {model_name}"
        (iw, ih), _ = cv2.getTextSize(info_str, font, 0.5, 1)
        cv2.rectangle(annotated, (w - iw - 25, 15), (w - 15, 48), (7, 16, 17), -1)
        cv2.rectangle(annotated, (w - iw - 25, 15), (w - 15, 48), (27, 50, 52), 1)
        cv2.putText(annotated, info_str, (w - iw - 18, 38), font, 0.5, (247, 250, 245), 1, cv2.LINE_AA)

        # 4. Bottom Status Overlay Bar
        is_acc = analysis.is_accident
        bar_color = (0, 0, 180) if is_acc else (7, 16, 17)
        text_color = (255, 255, 255) if is_acc else (157, 245, 81)
        status_text = f"STATUS: ACCIDENT DETECTED (SEVERITY: {analysis.severity.level})" if is_acc else "STATUS: NORMAL"

        cv2.rectangle(annotated, (0, h - 45), (w, h), bar_color, -1)
        cv2.putText(annotated, status_text, (20, h - 15), font, 0.6, text_color, 2 if is_acc else 1, cv2.LINE_AA)

        if incident_id:
            id_text = f"INCIDENT: {incident_id}"
            (idw, _), _ = cv2.getTextSize(id_text, font, 0.5, 1)
            cv2.putText(annotated, id_text, (w - idw - 20, h - 15), font, 0.5, (214, 255, 203), 1, cv2.LINE_AA)

        # Resize to 1280x720 while maintaining aspect ratio
        display_frame = self.letterbox_resize(annotated, settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT)
        return display_frame
