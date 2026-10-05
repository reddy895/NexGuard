"""
NexGuard — OpenCV Visualization Layer
=======================================
Draws real-time overlays on the video frame using OpenCV.
Runs as a cv2.imshow() window from the terminal — no web UI needed.

Color scheme:
    Person tracks  : GREEN  (0, 220, 80)
    Vehicle tracks : CYAN   (0, 210, 255)
    Accident zone  : RED    (0, 0, 255)
    WARNING        : ORANGE (0, 140, 255)
    CRITICAL       : DARK RED (0, 0, 200) flashing
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from nexguard.models.schemas import Incident, IncidentStatus, Severity, Track

# Colors (BGR)
COLOR_PERSON = (0, 220, 80)
COLOR_VEHICLE = (0, 210, 255)
COLOR_ACCIDENT = (0, 0, 255)
COLOR_WARNING = (0, 140, 255)
COLOR_CRITICAL = (0, 0, 180)
COLOR_NORMAL = (160, 160, 160)
COLOR_TEXT = (240, 240, 240)
COLOR_BG = (15, 18, 24)
COLOR_TEAL = (0, 230, 180)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MED = 0.55
FONT_LARGE = 0.70


class Visualizer:
    """
    Draws all NexGuard overlays onto the current video frame.

    Call draw_frame() each loop iteration with the current frame,
    tracks, and incident state to get back a fully annotated frame.
    """

    def __init__(self, window_name: str = "NexGuard — AI Safety Monitoring") -> None:
        self.window_name = window_name
        self._flash_state = False
        self._last_flash = time.time()
        self._frame_count = 0

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        fps: float,
        frame_index: int,
        active_incidents: Optional[List[Incident]] = None,
        whatsapp_status: str = "OFFLINE",
    ) -> np.ndarray:
        """
        Draws all overlays and returns the annotated frame.

        Args:
            frame:            BGR frame from OpenCV.
            tracks:           Current active tracks.
            fps:              Measured frames-per-second.
            frame_index:      Current frame number.
            active_incidents: List of active incident objects.
            whatsapp_status:  Current WhatsApp bot status string.

        Returns:
            Annotated BGR frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        active_incidents = active_incidents or []

        is_accident = any(
            i.status in (IncidentStatus.CONFIRMED, IncidentStatus.MONITORING, IncidentStatus.ALERT_SENT)
            for i in active_incidents
        )

        # Flash timer
        now = time.time()
        if now - self._last_flash > 0.4:
            self._flash_state = not self._flash_state
            self._last_flash = now

        # ── Draw tracks ──────────────────────────────────────────────────────
        persons, vehicles = 0, 0
        for track in tracks:
            if track.is_person:
                persons += 1
                color = COLOR_PERSON
            else:
                vehicles += 1
                color = COLOR_VEHICLE

            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"#{track.track_id} {track.class_name} {track.speed:.1f}px/f"
            lx, ly = x1, max(14, y1 - 6)
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SMALL, 1)
            cv2.rectangle(canvas, (lx - 1, ly - th - 3), (lx + tw + 1, ly + 2), (20, 22, 28), -1)
            cv2.putText(canvas, label, (lx, ly), FONT, FONT_SMALL, color, 1, cv2.LINE_AA)

            # Trajectory trail
            for pt in track.history[:-1]:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, color, -1)

            # Motion arrow
            if len(track.history) >= 3:
                p1 = (int(track.history[-3][0]), int(track.history[-3][1]))
                p2 = (int(track.cx), int(track.cy))
                if p1 != p2:
                    cv2.arrowedLine(canvas, p1, p2, color, 1, tipLength=0.35)

        # ── Accident overlay ─────────────────────────────────────────────────
        if is_accident and active_incidents:
            inc = active_incidents[0]
            cx = int(inc.location.get("cx", w / 2))
            cy = int(inc.location.get("cy", h / 2))
            radius = 80

            # Pulsing circle
            pulse_r = radius + (10 if self._flash_state else 0)
            overlay = canvas.copy()
            cv2.circle(overlay, (cx, cy), pulse_r, COLOR_ACCIDENT, -1)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            cv2.circle(canvas, (cx, cy), pulse_r, COLOR_ACCIDENT, 2)
            cv2.putText(canvas, "COLLISION ZONE", (cx - 55, cy),
                        FONT, FONT_SMALL, (255, 255, 255), 1, cv2.LINE_AA)

            # Severity banner
            sev = inc.severity.value
            banner_color = {
                "LOW": COLOR_WARNING,
                "MEDIUM": (0, 100, 220),
                "HIGH": COLOR_ACCIDENT,
                "CRITICAL": COLOR_CRITICAL if self._flash_state else (0, 0, 120),
            }.get(sev, COLOR_ACCIDENT)

            bx1, by1 = w // 2 - 200, 60
            bx2, by2 = w // 2 + 200, 110
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), banner_color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            banner_text = f"ACCIDENT CONFIRMED — {sev}"
            (btw, _), _ = cv2.getTextSize(banner_text, FONT, FONT_MED, 2)
            cv2.putText(canvas, banner_text,
                        (w // 2 - btw // 2, by1 + 32),
                        FONT, FONT_MED, (255, 255, 255), 2, cv2.LINE_AA)
            conf_text = f"Confidence: {inc.confidence:.0%}   ID: {inc.incident_id}"
            (ctw, _), _ = cv2.getTextSize(conf_text, FONT, FONT_SMALL, 1)
            cv2.putText(canvas, conf_text,
                        (w // 2 - ctw // 2, by1 + 52),
                        FONT, FONT_SMALL, (220, 220, 220), 1, cv2.LINE_AA)

        # ── Top HUD bar ──────────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (w, 44), (12, 15, 20), -1)
        cv2.line(canvas, (0, 44), (w, 44), COLOR_TEAL, 1)

        cv2.putText(canvas, "NEXGUARD AI SURVEILLANCE", (12, 28),
                    FONT, FONT_MED, COLOR_TEAL, 2, cv2.LINE_AA)

        # FPS badge
        fps_color = (0, 200, 100) if fps >= 10 else COLOR_WARNING
        cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 140, 28),
                    FONT, FONT_MED, fps_color, 1, cv2.LINE_AA)

        # WhatsApp badge
        wa_color = (0, 200, 80) if whatsapp_status == "READY" else COLOR_WARNING
        wa_text = f"WA: {whatsapp_status}"
        (wtw, _), _ = cv2.getTextSize(wa_text, FONT, FONT_SMALL, 1)
        cv2.putText(canvas, wa_text, (w - 200, 12),
                    FONT, FONT_SMALL, wa_color, 1, cv2.LINE_AA)

        # ── Bottom stats bar ─────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, h - 36), (w, h), (12, 15, 20), -1)
        cv2.line(canvas, (0, h - 36), (w, h - 36), (60, 60, 60), 1)

        stats = (
            f"Frame: {frame_index}   "
            f"Persons: {persons}   "
            f"Vehicles: {vehicles}   "
            f"Tracks: {len(tracks)}   "
            f"Incidents: {len(active_incidents)}"
        )
        cv2.putText(canvas, stats, (10, h - 12),
                    FONT, FONT_SMALL, COLOR_NORMAL, 1, cv2.LINE_AA)

        # Accident status indicator (right side of bottom bar)
        if is_accident:
            status_color = COLOR_ACCIDENT if self._flash_state else (100, 0, 0)
            cv2.putText(canvas, "● INCIDENT ACTIVE", (w - 200, h - 12),
                        FONT, FONT_SMALL, status_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "● MONITORING", (w - 170, h - 12),
                        FONT, FONT_SMALL, (0, 160, 60), 1, cv2.LINE_AA)

        return canvas

    def show(self, frame: np.ndarray) -> int:
        """
        Displays the frame in the OpenCV window.

        Returns:
            Key code from cv2.waitKey(1) — check for 'q'/ESC to quit.
        """
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def destroy(self) -> None:
        """Closes the OpenCV window."""
        cv2.destroyAllWindows()
