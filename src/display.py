"""
NexGuard Visual Display & HUD Rendering Module
Renders bounding boxes, involved vehicle RED highlighting, visual overlays,
FPS counters, status headers, and alert banners onto video frames.
"""

from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np

from config import config
from src.detector import DetectionObject
from src.tracker import TrackedObject
from src.accident_detector import AccidentAnalysisResult, AccidentState
from src.severity_engine import SeverityEvaluation


class DisplayRenderer:
    """Renders NexGuard visual elements and overlays onto OpenCV frames."""

    def __init__(self, window_name: str = None):
        self.window_name = window_name or config.window_name

    def init_window(self):
        """Initializes OpenCV GUI window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, config.display_width, config.display_height)

    def draw_hud(
        self,
        frame: np.ndarray,
        detections: List[DetectionObject],
        tracked_objects: List[TrackedObject],
        analysis: AccidentAnalysisResult,
        severity: SeverityEvaluation,
        fps: float,
        model_name: str,
        whatsapp_status: str,
        incident_id: Optional[str] = None
    ) -> np.ndarray:
        """Draws bounding boxes, RED accident highlights, and top/bottom overlays onto frame."""
        if frame is None:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        involved_set = set(analysis.involved_track_ids) if analysis.is_accident else set()

        # Step 1: Draw Bounding Boxes
        for track in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            is_involved = (track.track_id in involved_set)

            if is_involved:
                # ACCIDENT VEHICLE: STRONG RED (BGR: 0, 0, 255)
                color = (0, 0, 255)
                thickness = 4
                label = f"{track.class_name.upper()} #{track.track_id} ACCIDENT"
            elif track.class_name == "person":
                color = (255, 144, 30)  # Bright Blue/Cyan
                thickness = 2
                label = f"PERSON #{track.track_id}"
            elif track.class_name in ["car", "motorcycle"]:
                color = (0, 230, 0)  # Green
                thickness = 2
                label = f"{track.class_name.upper()} #{track.track_id}"
            elif track.class_name in ["bus", "truck"]:
                color = (0, 215, 255)  # Gold/Yellow
                thickness = 2
                label = f"{track.class_name.upper()} #{track.track_id}"
            else:
                color = (200, 200, 200)
                thickness = 2
                label = f"{track.class_name.upper()} #{track.track_id}"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label banner
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - lbl_h - 6)), (x1 + lbl_w + 6, max(lbl_h + 6, y1)), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(lbl_h + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if is_involved else (0, 0, 0),
                2
            )

            # Draw track history trail
            if len(track.center_history) > 1:
                pts = np.array([(int(cx), int(cy)) for cx, cy in track.center_history], np.int32)
                cv2.polylines(annotated, [pts], False, color, 2)

        # Step 2: Draw Top Overlay Header
        header_bg = np.zeros((55, w, 3), dtype=np.uint8)
        alpha = 0.65
        annotated[0:55, 0:w] = cv2.addWeighted(annotated[0:55, 0:w], 1 - alpha, header_bg, alpha, 0)

        # Left Header Text
        title_text = "NEXGUARD LIVE EDGE AI"
        cv2.putText(annotated, title_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Right Header Info
        info_text = f"FPS: {fps:.1f} | MODEL: {model_name} | DEVICE: {config.device.upper()}"
        (iw, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(annotated, info_text, (w - iw - 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Step 3: Draw Bottom Status Banner
        banner_h = 70
        banner_bg = np.zeros((banner_h, w, 3), dtype=np.uint8)
        
        if analysis.is_accident:
            # RED banner for active accident
            banner_bg[:, :] = (0, 0, 180)
            status_str = "STATUS: ACCIDENT DETECTED"
            status_color = (255, 255, 255)
        else:
            banner_bg[:, :] = (30, 30, 30)
            status_str = f"STATUS: {analysis.state.value}"
            status_color = (0, 255, 0)

        annotated[h - banner_h:h, 0:w] = cv2.addWeighted(annotated[h - banner_h:h, 0:w], 0.3, banner_bg, 0.7, 0)

        # Status text line 1
        cv2.putText(annotated, status_str, (15, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        # Status text line 2 (Severity, Involved vehicles, Person involvement)
        if analysis.is_accident:
            inv_str = ", ".join([f"#{tid}" for tid in analysis.involved_track_ids]) or "N/A"
            person_str = f"DETECTED ({analysis.people_near_accident})" if analysis.people_near_accident > 0 else "NONE"
            details = f"SEVERITY: {severity.level} | INVOLVED: {inv_str} | PERSON INVOLVEMENT: {person_str} | WA: {whatsapp_status}"
            cv2.putText(annotated, details, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            details = f"SEVERITY: {severity.level} | WHATSAPP: {whatsapp_status} | MODE: {config.performance_mode.upper()}"
            cv2.putText(annotated, details, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        return annotated
