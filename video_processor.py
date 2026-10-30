"""
NexGuard — Video Processing Pipeline
======================================
Handles frame-by-frame processing for both live webcam feeds
and local video file playback.

All processing is local — no frames are uploaded to any server.
Tkinter's native file dialog is used only for file selection;
no Tkinter application or GUI is built.

Usage:
    processor = VideoProcessor(detector, tracker, accident_detector,
                               severity_classifier, incident_manager, overlay)
    processor.run_webcam(camera_index=0)
    processor.run_file(file_path="/path/to/video.mp4")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import cfg
from nexguard.accident.accident_detector import AccidentDetector
from nexguard.accident.event import IncidentManager
from nexguard.accident.severity import SeverityClassifier, analyze_person_involvement
from nexguard.detection.detector import ObjectDetector
from nexguard.models.schemas import Severity
from nexguard.tracking.tracker import ObjectTracker
from nexguard.utils.logging import get_logger
from nexguard.utils.utils import FPSCounter, seconds_to_timestamp
from overlay import Overlay

log = get_logger("nexguard.video_processor")

# Keys that stop playback
_QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q, Q, ESC


def pick_video_file() -> Optional[str]:
    """
    Opens the OS native file picker (Tkinter) to select a video file.
    Returns the selected path, or None if the user cancelled.
    Does not build a Tkinter GUI — only the file dialog is used.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()   # hide the root window
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="NexGuard — Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path if path else None
    except Exception as exc:
        log.error("File dialog failed: %s", exc)
        print(f"[ERROR] Could not open file picker: {exc}")
        print("[INFO] Enter video path manually:")
        manual = input("  Path: ").strip()
        return manual if manual else None


class VideoProcessor:
    """
    Orchestrates the full detection pipeline for a single video source.

    Components injected at construction time so they can be shared
    and reset cleanly across multiple run_* calls.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        tracker: ObjectTracker,
        accident_detector: AccidentDetector,
        severity_classifier: SeverityClassifier,
        incident_manager: IncidentManager,
        overlay: Overlay,
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._acc_detector = accident_detector
        self._severity = severity_classifier
        self._incidents = incident_manager
        self._overlay = overlay

        diag = self._detector.get_diagnostic_info()
        log.info(
            "VideoProcessor initialized | Model: %s (%s) | Device: %s | Conf Thresh: %.2f | ImgSz: %d",
            diag.model_path, diag.model_type.value, diag.device, diag.confidence_threshold, diag.inference_size,
        )

    # ── Public entry points ────────────────────────────────────────────────

    def run_webcam(self, camera_index: int = 0) -> None:
        """
        Opens the webcam and runs live detection until Q is pressed.

        Args:
            camera_index: cv2.VideoCapture device index (default 0).
        """
        print(f"\n[INFO] Opening webcam (device {camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(
                f"[ERROR] Cannot open webcam at index {camera_index}.\n"
                "  • Check that a camera is physically connected.\n"
                "  • Try a different device index (e.g., 1, 2).\n"
                "  • Ensure no other application is holding the camera."
            )
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Webcam opened — {w}×{h}")
        print("[INFO] Press Q or ESC in the video window to stop.\n")

        self._reset_state()
        self._run_loop(cap, is_live=True, source_fps=30.0)

        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Webcam released. Returning to menu.")

    def run_file(self, file_path: str, loop: bool = True) -> None:
        """
        Processes a local video file frame-by-frame.

        Args:
            file_path: Absolute or relative path to the video file.
            loop: Whether to loop the video continuously until user exits (default True).
        """
        path = Path(file_path)
        if not path.exists():
            print(f"[ERROR] File not found: {file_path}")
            return

        print(f"\n[INFO] Opening video: {path.name}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(
                f"[ERROR] Cannot open video file: {file_path}\n"
                "  The file may be corrupt, empty, or in an unsupported format."
            )
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / source_fps if source_fps > 0 else 0
        print(
            f"[INFO] {w}×{h} @ {source_fps:.1f} FPS — "
            f"{total_frames} frames ({duration_s:.1f}s)"
        )
        print("[INFO] Continuous playback active — press Q or ESC in the video window to stop.\n")

        self._reset_state()
        self._run_loop(cap, is_live=False, source_fps=source_fps, total_frames=total_frames, loop=loop)

        cap.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Playback finished. Returning to menu.")

    # ── Core processing loop ───────────────────────────────────────────────

    def _run_loop(
        self,
        cap: cv2.VideoCapture,
        *,
        is_live: bool,
        source_fps: float,
        total_frames: int = 0,
        loop: bool = False,
    ) -> None:
        """
        Main frame processing loop shared by webcam and file modes.

        Exits when:
        - Q/ESC is pressed in the OpenCV window
        - The video file ends (if loop=False or file cannot be rewound)
        - The camera is disconnected
        """
        fps_counter = FPSCounter()
        frame_index: int = 0
        current_severity: Severity = Severity.NO_ACCIDENT

        try:
            while True:
                if is_live:
                    # Flush buffered frames for live sources to reduce latency
                    for _ in range(2):
                        cap.grab()
                    ok, frame = cap.retrieve()
                else:
                    ok, frame = cap.read()

                if not ok:
                    if not is_live and loop:
                        # Video file reached end — rewind to beginning for continuous loop
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            log.info("Video file ended and could not be re-read.")
                            break
                        # Reset tracking state for seamless loop restart
                        self._tracker.reset()
                        self._acc_detector.reset()
                    else:
                        if not is_live:
                            log.info("Video file ended at frame %d.", frame_index)
                        break

                frame_index += 1
                fps_counter.tick()

                # ── Detection ───────────────────────────────────────────
                detections = self._detector.detect(frame, frame_number=frame_index)

                # ── Tracking ────────────────────────────────────────────
                tracks = self._tracker.update(detections)
                persons  = [t for t in tracks if t.is_person]
                vehicles = [t for t in tracks if t.is_vehicle]

                # ── Incident housekeeping ────────────────────────────────
                self._incidents.update_frame(frame_index)
                self._incidents.resolve_stale_incidents([t.track_id for t in tracks])

                # ── Accident detection ───────────────────────────────────
                accident_event = self._acc_detector.update(tracks)

                if accident_event:
                    person_inv = analyze_person_involvement(persons, accident_event)
                    involved_tracks = [t for t in tracks if t.track_id in accident_event.track_ids]
                    severity_result = self._severity.classify(
                        event=accident_event,
                        involved_tracks=involved_tracks,
                        all_tracks=tracks,
                        person_involvements=person_inv,
                        persistence_frames=self._acc_detector.consecutive_count,
                    )
                    incident = self._incidents.create_incident(
                        event=accident_event,
                        severity_result=severity_result,
                        involved_tracks=involved_tracks,
                        person_involvements=person_inv,
                    )
                    current_severity = severity_result.severity
                    self._print_incident_summary(incident, frame_index)

                # Update severity from active incidents
                active = self._incidents.active_incidents
                if active:
                    # Use highest severity among active incidents
                    sev_order = [
                        Severity.NO_ACCIDENT, Severity.POSSIBLE,
                        Severity.LOW, Severity.MEDIUM,
                        Severity.HIGH, Severity.UNCERTAIN,
                    ]
                    current_severity = max(
                        (inc.severity for inc in active),
                        key=lambda s: sev_order.index(s) if s in sev_order else 0,
                        default=Severity.NO_ACCIDENT,
                    )
                elif accident_event is None:
                    # Decay toward NO_ACCIDENT when no active incidents
                    if current_severity != Severity.NO_ACCIDENT:
                        # Keep POSSIBLE visible for a few frames
                        if frame_index % 10 == 0:
                            current_severity = Severity.NO_ACCIDENT

                # ── Compute video timestamp ──────────────────────────────
                timestamp_s = 0.0
                if not is_live and source_fps > 0:
                    timestamp_s = frame_index / source_fps

                # ── Terminal progress (every 30 frames) ──────────────────
                if frame_index % 30 == 0:
                    self._print_progress(
                        frame_index, total_frames, fps_counter.fps,
                        len(persons), len(vehicles), len(active), timestamp_s, is_live
                    )

                # ── OpenCV display ───────────────────────────────────────
                annotated = self._overlay.draw(
                    frame=frame,
                    tracks=tracks,
                    fps=fps_counter.fps,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    active_incidents=active,
                    current_severity=current_severity,
                )
                key = self._overlay.show(annotated)
                if key in _QUIT_KEYS:
                    print("\n[INFO] Q pressed — stopping.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted (Ctrl+C).")

    # ── State management ───────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Resets all stateful components before starting a new source."""
        self._tracker.reset()
        self._acc_detector.reset()
        # IncidentManager: clear previous session incidents
        self._incidents.reset()

    # ── Terminal output ────────────────────────────────────────────────────

    @staticmethod
    def _print_progress(
        frame: int,
        total: int,
        fps: float,
        persons: int,
        vehicles: int,
        incidents: int,
        ts: float,
        is_live: bool,
    ) -> None:
        progress = f"{frame}/{total}" if total > 0 else str(frame)
        ts_str = "LIVE" if is_live else seconds_to_timestamp(ts)
        print(
            f"\r  Frame {progress:>10} | {ts_str} | "
            f"FPS {fps:5.1f} | "
            f"P {persons} V {vehicles} | "
            f"Incidents {incidents}",
            end="",
            flush=True,
        )

    @staticmethod
    def _print_incident_summary(incident, frame_index: int) -> None:
        print(f"\n\n{'─' * 60}")
        print(f"  [INCIDENT] {incident.incident_id}")
        print(f"  Severity   : {incident.severity.value}")
        print(f"  Confidence : {incident.confidence:.0%}")
        print(f"  Type       : {incident.event_type.value}")
        print(f"  Tracks     : {incident.involved_tracks}")
        print(f"  Frame      : {frame_index}")
        for reason in incident.severity_reasons[:4]:
            print(f"             • {reason}")
        print(f"{'─' * 60}\n")
