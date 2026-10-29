"""
NexGuard — OpenCV Visualization Overlay
=========================================
Draws all real-time overlays on the video frame using OpenCV.
Runs purely as a cv2.imshow() window — no web UI, no browser.

Color scheme (BGR):
    Person tracks      : GREEN   (50, 220, 80)
    Car tracks         : CYAN    (0, 210, 255)
    Motorcycle tracks  : MAGENTA (200, 50, 200)
    Bus tracks         : ORANGE  (0, 165, 255)
    Truck tracks       : TEAL    (180, 210, 80)
    Bicycle tracks     : YELLOW  (0, 230, 200)
    Possible accident  : AMBER   (0, 180, 255)
    LOW severity       : AMBER   (0, 190, 240)
    MEDIUM severity    : ORANGE  (0, 130, 255)
    HIGH severity      : RED     (0, 50, 255)
    UNCERTAIN severity : GREY    (180, 180, 180)
"""

from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from nexguard.models.schemas import Incident, IncidentStatus, Severity, Track

# ---------------------------------------------------------------------------
# Palette (BGR)
# ---------------------------------------------------------------------------
_C_PERSON     = (50, 220, 80)
_C_CAR        = (0, 210, 255)
_C_MOTO       = (200, 50, 200)
_C_BUS        = (0, 165, 255)
_C_TRUCK      = (180, 210, 80)
_C_BICYCLE    = (0, 230, 200)
_C_DEFAULT_V  = (0, 210, 255)   # fallback vehicle color
_C_POSSIBLE   = (0, 180, 255)   # amber
_C_LOW        = (0, 190, 240)   # amber-yellow
_C_MEDIUM     = (0, 130, 255)   # orange
_C_HIGH       = (0, 50, 255)    # red
_C_UNCERTAIN  = (180, 180, 180) # grey
_C_NONE       = (80, 200, 80)   # calm green when no incident
_C_TEXT       = (240, 240, 240)
_C_HUD_BG     = (14, 17, 22)
_C_ACCENT     = (0, 230, 180)   # teal

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FS_SMALL   = 0.42
FS_MED     = 0.52
FS_LARGE   = 0.68
FS_XLARGE  = 0.80

_VEHICLE_COLORS = {
    "car":        _C_CAR,
    "motorcycle": _C_MOTO,
    "bus":        _C_BUS,
    "truck":      _C_TRUCK,
    "bicycle":    _C_BICYCLE,
}

_SEVERITY_COLORS = {
    Severity.NO_ACCIDENT: _C_NONE,
    Severity.POSSIBLE:    _C_POSSIBLE,
    Severity.LOW:         _C_LOW,
    Severity.MEDIUM:      _C_MEDIUM,
    Severity.HIGH:        _C_HIGH,
    Severity.UNCERTAIN:   _C_UNCERTAIN,
}

_SEVERITY_LABELS = {
    Severity.NO_ACCIDENT: "NO ACCIDENT DETECTED",
    Severity.POSSIBLE:    "POSSIBLE ACCIDENT",
    Severity.LOW:         "LOW SEVERITY — LOW-IMPACT COLLISION SUSPECTED",
    Severity.MEDIUM:      "MEDIUM SEVERITY — POSSIBLE PERSON FALL / SIGNIFICANT IMPACT",
    Severity.HIGH:        "HIGH SEVERITY — MAJOR INCIDENT SUSPECTED",
    Severity.UNCERTAIN:   "UNCERTAIN — MANUAL REVIEW REQUIRED",
}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_text_bg(
    img: np.ndarray,
    text: str,
    origin: tuple,
    font_scale: float,
    color: tuple,
    thickness: int = 1,
    bg: tuple = (10, 12, 18),
    padding: int = 3,
) -> None:
    """Draws text with a dark background rectangle for readability."""
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    cv2.rectangle(
        img,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg, -1,
    )
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_filled_rect(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    alpha: float = 1.0,
) -> None:
    """Draws a filled rectangle, optionally semi-transparent."""
    if alpha >= 1.0:
        cv2.rectangle(img, pt1, pt2, color, -1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


# ---------------------------------------------------------------------------
# Main overlay class
# ---------------------------------------------------------------------------

class Overlay:
    """
    Draws all NexGuard overlays onto the current video frame.

    Call draw() each loop iteration with the current frame, tracks,
    and incident state to get back a fully annotated BGR frame.
    """

    def __init__(self, window_name: str = "NexGuard — AI Road Accident Detection") -> None:
        self.window_name = window_name
        self._flash_on = False
        self._last_flash = time.time()
        self._event_counter: int = 0

    # ── Frame draw ─────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        fps: float,
        frame_index: int,
        timestamp_s: float = 0.0,
        active_incidents: Optional[List[Incident]] = None,
        current_severity: Severity = Severity.NO_ACCIDENT,
    ) -> np.ndarray:
        """
        Draws all overlays and returns the annotated frame.

        Args:
            frame:            BGR frame from OpenCV.
            tracks:           Current active tracks.
            fps:              Measured frames-per-second.
            frame_index:      Current frame number.
            timestamp_s:      Video position in seconds (0 for webcam).
            active_incidents: List of active Incident objects.
            current_severity: Current severity state to display.

        Returns:
            Annotated BGR frame (copy of input).
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        active_incidents = active_incidents or []

        # Flash toggle (0.4 s period)
        now = time.time()
        if now - self._last_flash > 0.4:
            self._flash_on = not self._flash_on
            self._last_flash = now

        is_incident = current_severity not in (Severity.NO_ACCIDENT,)

        # ── 1. Draw tracked objects ───────────────────────────────────────
        n_persons, n_vehicles = 0, 0
        involved_ids: set = set()
        for inc in active_incidents:
            involved_ids.update(inc.involved_tracks)

        for track in tracks:
            if track.is_person:
                color = _C_PERSON
                n_persons += 1
            else:
                color = _VEHICLE_COLORS.get(track.class_name, _C_DEFAULT_V)
                n_vehicles += 1

            x1, y1, x2, y2 = map(int, track.bbox)

            # Highlight involved tracks in red during an incident
            if track.track_id in involved_ids and is_incident:
                sev_color = _SEVERITY_COLORS.get(current_severity, _C_HIGH)
                thickness = 3 if self._flash_on else 2
                cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), sev_color, thickness)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label: class + confidence
            label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
            _put_text_bg(canvas, label, (x1, max(14, y1 - 4)), FS_SMALL, color)

            # Trajectory trail dots
            for pt in track.history[:-1]:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, color, -1)

            # Motion arrow (last 3 trajectory points)
            if len(track.history) >= 3:
                p1 = (int(track.history[-3][0]), int(track.history[-3][1]))
                p2 = (int(track.cx), int(track.cy))
                if p1 != p2:
                    cv2.arrowedLine(canvas, p1, p2, color, 1, tipLength=0.35)

        # ── 2. Collision zone overlay ─────────────────────────────────────
        if is_incident and active_incidents:
            inc = active_incidents[0]
            icx = int(inc.location.get("cx", w / 2))
            icy = int(inc.location.get("cy", h / 2))
            sev_color = _SEVERITY_COLORS.get(current_severity, _C_HIGH)

            pulse_r = 75 + (12 if self._flash_on else 0)
            overlay = canvas.copy()
            cv2.circle(overlay, (icx, icy), pulse_r, sev_color, -1)
            cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
            cv2.circle(canvas, (icx, icy), pulse_r, sev_color, 2)
            _put_text_bg(canvas, "INCIDENT ZONE", (icx - 50, icy + 6),
                         FS_SMALL, (255, 255, 255), bg=(0, 0, 0))

        # ── 3. Top HUD bar ────────────────────────────────────────────────
        _draw_filled_rect(canvas, (0, 0), (w, 42), _C_HUD_BG)
        cv2.line(canvas, (0, 42), (w, 42), _C_ACCENT, 1)
        cv2.putText(canvas, "NEXGUARD AI", (10, 28), FONT, FS_LARGE, _C_ACCENT, 2, cv2.LINE_AA)

        fps_col = (0, 200, 100) if fps >= 10 else _C_POSSIBLE
        cv2.putText(canvas, f"FPS {fps:.1f}", (w - 110, 28), FONT, FS_MED, fps_col, 1, cv2.LINE_AA)

        # ── 4. Upper-right severity overlay ──────────────────────────────
        self._draw_severity_panel(canvas, current_severity, active_incidents, timestamp_s)

        # ── 5. Bottom stats bar ───────────────────────────────────────────
        _draw_filled_rect(canvas, (0, h - 34), (w, h), _C_HUD_BG)
        cv2.line(canvas, (0, h - 34), (w, h - 34), (50, 50, 50), 1)
        stats = (
            f"Frame {frame_index}  |  "
            f"Persons {n_persons}  Vehicles {n_vehicles}  "
            f"Tracks {len(tracks)}  |  Incidents {len(active_incidents)}"
        )
        cv2.putText(canvas, stats, (10, h - 10), FONT, FS_SMALL, (140, 140, 140), 1, cv2.LINE_AA)

        # Active incident indicator (bottom-right)
        if is_incident:
            ind_col = _C_HIGH if self._flash_on else (80, 20, 120)
            cv2.putText(canvas, "● INCIDENT ACTIVE", (w - 200, h - 10),
                        FONT, FS_SMALL, ind_col, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "● MONITORING", (w - 160, h - 10),
                        FONT, FS_SMALL, (0, 160, 60), 1, cv2.LINE_AA)

        return canvas

    # ── Severity panel ─────────────────────────────────────────────────────

    def _draw_severity_panel(
        self,
        canvas: np.ndarray,
        severity: Severity,
        incidents: List[Incident],
        timestamp_s: float,
    ) -> None:
        """Draws the compact severity status panel in the upper-right corner."""
        h, w = canvas.shape[:2]

        panel_w = 310
        panel_h = 100 if not incidents else 130
        x1 = w - panel_w - 8
        y1 = 50
        x2 = w - 8
        y2 = y1 + panel_h

        # Panel background
        _draw_filled_rect(canvas, (x1, y1), (x2, y2), (10, 12, 18), alpha=0.85)
        sev_color = _SEVERITY_COLORS.get(severity, _C_NONE)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), sev_color, 1)

        # "NEXGUARD" header
        cv2.putText(canvas, "NEXGUARD", (x1 + 8, y1 + 18),
                    FONT, FS_SMALL, _C_ACCENT, 1, cv2.LINE_AA)

        # STATUS line
        status_label = _SEVERITY_LABELS.get(severity, "UNKNOWN")
        # Truncate if too long for the panel
        if len(status_label) > 38:
            status_label = status_label[:35] + "..."
        cv2.putText(canvas, f"STATUS: {severity.value}", (x1 + 8, y1 + 38),
                    FONT, FS_SMALL, sev_color, 1, cv2.LINE_AA)

        # Timestamp
        from nexguard.utils.utils import seconds_to_timestamp
        ts_str = seconds_to_timestamp(timestamp_s) if timestamp_s > 0 else "LIVE"
        cv2.putText(canvas, f"TIME: {ts_str}", (x1 + 8, y1 + 56),
                    FONT, FS_SMALL, _C_TEXT, 1, cv2.LINE_AA)

        # Incident-specific fields
        if incidents:
            inc = incidents[0]
            cv2.putText(canvas, f"EVENT: {inc.incident_id}", (x1 + 8, y1 + 74),
                        FONT, FS_SMALL, sev_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"CONF: {inc.confidence:.0%}", (x1 + 8, y1 + 92),
                        FONT, FS_SMALL, _C_TEXT, 1, cv2.LINE_AA)

            # Severity description (wrapped)
            short_label = _SEVERITY_LABELS.get(severity, "")[:42]
            cv2.putText(canvas, short_label, (x1 + 8, y1 + 114),
                        FONT, 0.36, sev_color, 1, cv2.LINE_AA)

    # ── Window management ──────────────────────────────────────────────────

    def show(self, frame: np.ndarray) -> int:
        """
        Displays the frame in the OpenCV window.

        Returns:
            Key code from cv2.waitKey(1). Check ord('q')/27 (ESC) to quit.
        """
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def destroy(self) -> None:
        """Closes all OpenCV windows."""
        cv2.destroyAllWindows()
