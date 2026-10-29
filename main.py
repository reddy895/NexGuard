#!/usr/bin/env python3
"""
NexGuard — AI Road Accident Detection System
=============================================
Terminal-based entry point.

Usage:
    python main.py

The menu provides:
  [1] Live Detection — System Camera
  [2] Detect Accident in a Video File
  [3] Exit

Press Q or ESC in the video window to stop detection and return here.
No internet connection, cloud service, or API key is required during detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency guard — give a clear message before any other import fails
# ---------------------------------------------------------------------------
def _check_deps() -> None:
    missing = []
    for pkg, imp in [("opencv-python", "cv2"), ("ultralytics", "ultralytics"), ("numpy", "numpy")]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("\n[ERROR] Missing required packages:")
        for p in missing:
            print(f"  pip install {p}")
        print("\nRun:  pip install -r requirements.txt\n")
        sys.exit(1)

_check_deps()

# Standard imports (after dep check)
import signal

# Local imports
from config import cfg
from nexguard.accident.accident_detector import AccidentDetector
from nexguard.accident.event import IncidentManager
from nexguard.accident.severity import SeverityClassifier
from nexguard.detection.detector import ObjectDetector
from nexguard.tracking.tracker import ObjectTracker
from nexguard.utils.logging import get_logger
from overlay import Overlay
from video_processor import VideoProcessor, pick_video_file

log = get_logger("nexguard.main")

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
_BANNER = """
==================================================
                 NEXGUARD AI
       ROAD ACCIDENT DETECTION SYSTEM
==================================================
"""

_MENU = """Select an operation:

  [1] Live Detection — System Camera
  [2] Detect Accident in a Video File
  [3] Exit

Enter your choice: """

# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------

def print_startup_info(detector: ObjectDetector) -> None:
    """Prints a startup summary to the terminal."""
    diag = detector.get_diagnostic_info()
    lines = [
        "──────────────────────────────────────────────────",
        "  NEXGUARD STARTUP DIAGNOSTICS",
        "──────────────────────────────────────────────────",
        f"  Model Path   : {diag.model_path}",
        f"  Device       : {diag.device.upper()}",
        f"  Model Type   : {diag.model_type.value.upper()}",
        f"  Classes      : {diag.class_count} total",
        f"  Conf Thresh  : {diag.confidence_threshold:.0%}",
    ]
    if diag.accident_classes_found:
        lines.append(f"  Accident Cls : {', '.join(diag.accident_classes_found)}")
    else:
        lines.append("  Accident Cls : None — using HEURISTIC collision detection")
    if not diag.ready and diag.error_message:
        lines.append(f"  [ERROR]      : {diag.error_message}")
    lines.append("──────────────────────────────────────────────────")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_pipeline() -> tuple:
    """
    Initialises and returns all pipeline components.
    Raises SystemExit if the model cannot be loaded.
    """
    print("[INFO] Loading YOLO model...")
    detector = ObjectDetector()
    if not detector.model_ready:
        diag = detector.get_diagnostic_info()
        print(f"\n[ERROR] Failed to load model: {diag.error_message}")
        print(
            "\nTroubleshooting:\n"
            "  1. Ensure 'ultralytics' is installed:  pip install ultralytics\n"
            "  2. Check NEXGUARD_MODEL_PATH in your .env or config.py\n"
            "  3. Ensure internet is available for first-time weight download\n"
        )
        sys.exit(1)

    print_startup_info(detector)

    tracker           = ObjectTracker()
    accident_detector = AccidentDetector()
    severity          = SeverityClassifier()
    incidents         = IncidentManager()
    overlay           = Overlay(window_name=cfg.window_name)

    processor = VideoProcessor(
        detector=detector,
        tracker=tracker,
        accident_detector=accident_detector,
        severity_classifier=severity,
        incident_manager=incidents,
        overlay=overlay,
    )
    return processor, incidents


def run() -> None:
    """Main application loop — displays menu and dispatches user choice."""
    print(_BANNER)

    processor, incidents = build_pipeline()

    _running = True

    def _sigint_handler(sig, frame):
        nonlocal _running
        print("\n\n[INFO] Ctrl+C received — exiting.\n")
        _running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    while _running:
        try:
            choice = input(_MENU).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            # ── Live webcam detection ────────────────────────────────────
            try:
                camera_index = int(
                    input("  Camera index [0]: ").strip() or "0"
                )
            except ValueError:
                camera_index = 0
            processor.run_webcam(camera_index=camera_index)

        elif choice == "2":
            # ── Video file detection ─────────────────────────────────────
            print("\n[INFO] Opening file picker...")
            file_path = pick_video_file()
            if not file_path:
                print("[INFO] No file selected — returning to menu.")
                continue
            processor.run_file(file_path=file_path)

        elif choice == "3":
            print("\n[INFO] Exiting NexGuard. Goodbye.\n")
            break

        else:
            print("  [!] Invalid choice — enter 1, 2, or 3.\n")

    # Session summary
    total = incidents.total_incidents
    if total > 0:
        print(f"\n  Session summary: {total} incident(s) detected.\n")


if __name__ == "__main__":
    run()
