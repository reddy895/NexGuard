#!/usr/bin/env python3
"""
NexGuard — AI-Powered CCTV Surveillance & Accident Response System
====================================================================
Main entry point. Runs entirely from the terminal.

Usage:
    python main.py                          # use default source (webcam 0)
    python main.py --source 0               # webcam
    python main.py --source test_clips/road.mp4  # video file (shows live)
    python main.py --source rtsp://...      # IP camera
    python main.py --no-display             # terminal-only (no OpenCV window)
    python main.py --no-whatsapp            # skip WhatsApp bot startup
    python main.py --help                   # show all options

Press Q or ESC in the video window to quit.
Press Ctrl+C in the terminal to quit.
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

# ── Local imports ─────────────────────────────────────────────────────────────
from config import cfg
from nexguard.accident.accident_detector import AccidentDetector
from nexguard.accident.event import IncidentManager
from nexguard.accident.severity import SeverityClassifier, analyze_person_involvement
from nexguard.alerts.manager import AlertManager
from nexguard.alerts.whatsapp import WhatsAppNotifier
from nexguard.detection.detector import ObjectDetector
from nexguard.evidence.recorder import EvidenceRecorder
from nexguard.input.source import VideoSource
from nexguard.models.schemas import IncidentStatus
from nexguard.tracking.tracker import ObjectTracker
from nexguard.utils.logging import get_logger
from nexguard.visualization.ui import Visualizer

log = get_logger("nexguard.main")

# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                        NEXGUARD                             ║
║              AI CCTV ACCIDENT DETECTION SYSTEM              ║
║                     Version 2.0.0                           ║
╚══════════════════════════════════════════════════════════════╝
"""

_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n\n[INFO] Shutdown signal received — stopping pipeline...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nexguard",
        description="NexGuard — AI-Powered CCTV Surveillance & Accident Response System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --source 0\n"
            "  python main.py --source test_clips/road.mp4\n"
            "  python main.py --source rtsp://192.168.1.100:554/stream\n"
            "  python main.py --no-display --no-whatsapp\n"
        ),
    )

    parser.add_argument(
        "--source",
        default=cfg.source,
        help="Video source: webcam index (0), file path, or RTSP URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=cfg.model_path,
        help="Path to YOLO model weights (default: %(default)s)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=cfg.confidence_threshold,
        help="Detection confidence threshold 0.0–1.0 (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=cfg.device,
        choices=["cpu", "cuda", "mps"],
        help="Inference device (default: %(default)s)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window — terminal output only",
    )
    parser.add_argument(
        "--no-whatsapp",
        action="store_true",
        help="Disable WhatsApp bot startup",
    )
    parser.add_argument(
        "--whatsapp-target",
        default="9591152862",
        help="WhatsApp target phone number (default: %(default)s)",
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        default=True,
        help="Save evidence frames and metadata on accident (default: enabled)",
    )
    parser.add_argument(
        "--accident-threshold",
        type=float,
        default=cfg.accident_threshold,
        help="Accident detection confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--max-fps",
        type=int,
        default=15,
        help="Limit pipeline speed to maximum FPS (default: %(default)s)",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def print_startup_info(args: argparse.Namespace, detector: ObjectDetector) -> None:
    print(BANNER)
    print("─" * 62)
    print(f"  [INFO] Model       : {args.model}")
    print(f"  [INFO] Device      : {args.device.upper()}")
    print(f"  [INFO] Source      : {args.source}")
    print(f"  [INFO] Confidence  : {args.conf:.0%}")
    print(f"  [INFO] Acc.Thresh  : {args.accident_threshold:.0%}")
    print(f"  [INFO] Display     : {'OFF' if args.no_display else 'OpenCV window'}")
    print(f"  [INFO] WhatsApp    : {'DISABLED' if args.no_whatsapp else 'STARTING (scan QR)'}")
    print(f"  [INFO] WA Target   : +{args.whatsapp_target}")
    print(f"  [INFO] Evidence    : {'ON' if args.save_evidence else 'OFF'}")
    print("─" * 62)
    if not args.no_display:
        print("  Press Q or ESC in the video window to quit.")
    print("  Press Ctrl+C in terminal to quit.")
    print("─" * 62 + "\n")


def print_frame_stats(
    frame_index: int,
    fps: float,
    n_persons: int,
    n_vehicles: int,
    n_incidents: int,
) -> None:
    """Prints a clean periodic terminal status line."""
    if frame_index % 30 == 0:  # Every ~1s at 30fps
        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"\r  [{ts}] Frame: {frame_index:6d} | "
            f"FPS: {fps:5.1f} | "
            f"Persons: {n_persons} | "
            f"Vehicles: {n_vehicles} | "
            f"Incidents: {n_incidents}",
            end="",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> int:
    """Main detection pipeline. Returns exit code."""
    global _running

    # ── Validate configuration ────────────────────────────────────────────────
    try:
        cfg.validate()
    except ValueError as e:
        # Only re-raise critical errors that would break the pipeline
        if "WHATSAPP" in str(e) and args.no_whatsapp:
            pass  # Fine — WhatsApp disabled anyway
        elif "WHATSAPP" not in str(e):
            log.error(f"Configuration error: {e}")
            return 1

    # ── Override config from CLI args ─────────────────────────────────────────
    cfg.source = args.source
    cfg.model_path = args.model
    cfg.confidence_threshold = args.conf
    cfg.device = args.device
    cfg.accident_threshold = args.accident_threshold
    cfg.whatsapp_enabled = not args.no_whatsapp

    # ── Initialize components ─────────────────────────────────────────────────
    log.info("Initializing NexGuard pipeline...")

    detector = ObjectDetector()
    if not detector.model_ready:
        log.error(
            "YOLO model failed to load. "
            "Ensure 'ultralytics' is installed and the model path is correct."
        )
        return 1

    tracker = ObjectTracker()
    accident_detector = AccidentDetector(threshold=args.accident_threshold)
    severity_classifier = SeverityClassifier()
    incident_manager = IncidentManager()
    evidence_recorder = EvidenceRecorder() if args.save_evidence else None

    # WhatsApp — starts Node.js bot in background (shows QR code if needed)
    notifier = WhatsAppNotifier(enabled=not args.no_whatsapp)
    alert_manager = AlertManager(notifier=notifier)

    # Override default target from CLI
    if args.whatsapp_target not in cfg.whatsapp_recipients:
        cfg.whatsapp_recipients = [args.whatsapp_target]

    visualizer = Visualizer() if not args.no_display else None

    print_startup_info(args, detector)

    # ── Open video source ─────────────────────────────────────────────────────
    try:
        source = VideoSource(args.source)
        source.open()
    except (FileNotFoundError, RuntimeError) as e:
        log.error(str(e))
        return 1

    log.info(
        f"Video: {source.frame_width}x{source.frame_height} @ {source.fps:.0f}fps  "
        f"[{source.source_type}]"
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_index = 0
    fps_start = time.time()
    fps_frames = 0
    measured_fps = 0.0
    evidence_recording: set = set()  # incident_ids with active clip recording

    print("\n[INFO] Pipeline running...\n")

    try:
        while _running and source.is_open():
            loop_start = time.time()
            ok, frame = source.read()
            if not ok:
                if source.source_type == "file":
                    log.info("Video file ended.")
                    break
                log.warning("Frame read failed — retrying...")
                time.sleep(0.05)
                continue

            frame_index += 1
            fps_frames += 1

            # FPS measurement (rolling window)
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                measured_fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            # ── Detection ──────────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Tracking ──────────────────────────────────────────────────────
            tracks = tracker.update(detections)

            persons = [t for t in tracks if t.is_person]
            vehicles = [t for t in tracks if t.is_vehicle]

            # ── Incident lifecycle: resolve stale ──────────────────────────────
            active_track_ids = [t.track_id for t in tracks]
            incident_manager.resolve_stale_incidents(active_track_ids)
            incident_manager.update_frame(frame_index)

            # ── Accident detection ────────────────────────────────────────────
            accident_event = accident_detector.update(tracks)

            if accident_event:
                # Person involvement analysis
                person_involvements = analyze_person_involvement(
                    persons, accident_event
                )

                # Get involved track objects
                involved_tracks = [
                    t for t in tracks if t.track_id in accident_event.track_ids
                ]

                # Severity classification
                severity_result = severity_classifier.classify(
                    event=accident_event,
                    involved_tracks=involved_tracks,
                    all_tracks=tracks,
                    person_involvements=person_involvements,
                    persistence_frames=accident_detector.consecutive_count,
                )

                # Create incident
                incident = incident_manager.create_incident(
                    event=accident_event,
                    severity_result=severity_result,
                    involved_tracks=involved_tracks,
                    person_involvements=person_involvements,
                )

                print(f"\n\n{'─'*62}")
                print(f"  [ACCIDENT] CONFIRMED — {incident.incident_id}")
                print(f"  Severity   : {incident.severity.value}")
                print(f"  Confidence : {incident.confidence:.0%}")
                print(f"  Type       : {incident.event_type.value}")
                print(f"  Tracks     : {incident.involved_tracks}")
                print(f"  People     : {len(incident.involved_people)}")
                if incident.severity_reasons:
                    for r in incident.severity_reasons[:3]:
                        print(f"              • {r}")
                print(f"{'─'*62}\n")

                # Evidence
                if evidence_recorder:
                    frame_path = evidence_recorder.save_frame(incident, frame, frame_index)
                    if frame_path:
                        incident_manager.add_evidence_frame(incident, frame_path)
                    evidence_recorder.save_metadata(incident)

                    if incident.incident_id not in evidence_recording:
                        if evidence_recorder.start_clip(incident, frame, fps=measured_fps or 15.0):
                            evidence_recording.add(incident.incident_id)

                # Alert dispatch
                if incident_manager.should_send_alert(incident):
                    print(f"  [ALERT] Dispatching WhatsApp alert...")
                    sent = alert_manager.dispatch(incident)
                    if sent:
                        incident_manager.mark_alert_sent(incident)
                        print(f"  [ALERT] Queued for delivery to +{args.whatsapp_target}\n")

            # ── Evidence clip frames for active incidents ──────────────────────
            if evidence_recorder:
                for inc in incident_manager.active_incidents:
                    if inc.incident_id in evidence_recording:
                        evidence_recorder.write_clip_frame(inc, frame)

            # ── Resolve clips for resolved incidents ───────────────────────────
            if evidence_recorder:
                for iid in list(evidence_recording):
                    resolved = all(
                        i.incident_id != iid
                        for i in incident_manager.active_incidents
                    )
                    if resolved:
                        clip = evidence_recorder.finalize_clip(
                            type("_FakeInc", (), {"incident_id": iid})()
                        )
                        evidence_recording.discard(iid)

            # ── WhatsApp status (cheap polling every 5s) ───────────────────────
            wa_status = "DISABLED"
            if not args.no_whatsapp and frame_index % 150 == 1:
                wa_status = notifier.get_status()
            elif not args.no_whatsapp:
                wa_status = getattr(notifier, "_last_wa_status", "LOADING")
            notifier._last_wa_status = wa_status  # type: ignore[attr-defined]

            # ── Terminal stats ────────────────────────────────────────────────
            print_frame_stats(
                frame_index,
                measured_fps,
                len(persons),
                len(vehicles),
                len(incident_manager.active_incidents),
            )

            # ── OpenCV display ────────────────────────────────────────────────
            if visualizer:
                annotated = visualizer.draw_frame(
                    frame=frame,
                    tracks=tracks,
                    fps=measured_fps,
                    frame_index=frame_index,
                    active_incidents=incident_manager.active_incidents,
                    whatsapp_status=wa_status,
                )
                key = visualizer.show(annotated)
                if key in (ord("q"), ord("Q"), 27):  # q or ESC
                    log.info("Quit key pressed — stopping.")
                    break

            # ── Frame Pacing (Max FPS) ────────────────────────────────────────
            if args.max_fps > 0:
                elapsed_loop = time.time() - loop_start
                target_time = 1.0 / args.max_fps
                if elapsed_loop < target_time:
                    time.sleep(target_time - elapsed_loop)

    except KeyboardInterrupt:
        pass
    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        print("\n")
        log.info("Shutting down pipeline...")

        source.release()

        if evidence_recorder:
            evidence_recorder.finalize_all()

        if visualizer:
            visualizer.destroy()

        alert_manager.shutdown()

        # Session summary
        print("\n" + "═" * 62)
        print("  NEXGUARD SESSION SUMMARY")
        print("═" * 62)
        print(f"  Frames processed : {frame_index}")
        print(f"  Total incidents  : {incident_manager.total_incidents}")
        print(f"  Resolved         : {len(incident_manager.incident_history)}")
        print(f"  Active           : {len(incident_manager.active_incidents)}")
        print("═" * 62 + "\n")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(args))
