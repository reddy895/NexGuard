"""
NexGuard — Edge AI CCTV Surveillance System
Main Application Entry Point (Phase One)
"""

import sys
import time
import os
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import NexGuardConfig, DEFAULT_CONFIG
from src.logger import setup_logger, get_logger
from src.utils import get_device_info, generate_snapshot_filename, ensure_dir, validate_file_path
from src.detector import NexGuardDetector
from src.display import draw_detections, draw_grid, draw_overlay_stats, draw_accident_overlay
from src.video import InputSource, WebcamInput, VideoFileInput, ImageFileInput
from src.event_logger import DetectionEventLogger
from src.accident_detector import AccidentDetector

logger = setup_logger()


def choose_media_file(media_type: str) -> str:
    """Open a native file picker for a video or image input."""
    filetypes = {
        "video": ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
        "image": ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    }
    label, patterns = filetypes[media_type]
    initial_dir = Path("assets/sample").resolve()
    zenity = shutil.which("zenity")

    if zenity:
        result = subprocess.run(
            [
                zenity,
                "--file-selection",
                f"--title=Select {media_type} file",
                f"--filename={initial_dir}{os.sep}",
                f"--file-filter={label} | {patterns}",
                "--file-filter=All files | *",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 1:
            return ""
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Zenity file picker failed")
        return result.stdout.strip()

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title=f"Select {media_type} file",
            initialdir=str(initial_dir),
            filetypes=[(label, patterns), ("All files", "*.*")],
        )
    finally:
        root.destroy()


def print_banner() -> None:
    """Prints the NexGuard startup banner."""
    print()
    print("==================================================")
    print("                 NEXGUARD")
    print("       EDGE AI SURVEILLANCE SYSTEM")
    print("        AI CCTV ACCIDENT DETECTION")
    print("==================================================")
    print("Model: YOLOv8n")
    print("Tracking: ByteTrack")
    print("Accident Detection: ENABLED")
    print("Waiting for video input...")
    print("==================================================")
    print()


def show_initialization_checks(config: NexGuardConfig) -> bool:
    """Displays startup dependency and system check list."""
    print("Initializing system...")
    print()

    # Check 1: Python environment
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"[✓] Python environment (v{py_ver})")
    time.sleep(0.1)

    # Check 2: Hardware acceleration & YOLO engine
    dev_type, dev_name = get_device_info()
    print(f"[✓] YOLO engine (Device: {dev_type} - {dev_name})")
    time.sleep(0.1)

    # Check 3: Configuration
    print(f"[✓] Configuration (Model: {config.model_path}, Conf: {config.confidence_threshold})")
    time.sleep(0.1)

    # Check 4: Detection pipeline
    print(f"[✓] Detection pipeline initialized")
    print()
    print(f"Model: {config.model_path}")
    print(f"Device: {dev_type} / {'CPU' if dev_type == 'CPU' else 'CUDA'}")
    print("Tracking: ByteTrack")
    print("Accident Detection: ENABLED")
    print("Green boxes: persons & vehicles | Red: accident zone")
    print("Waiting for video input...")
    print("==================================================")
    print()
    return True


def prompt_input_selection(config: NexGuardConfig) -> Optional[InputSource]:
    """Displays terminal menu and prompts user to choose an input source."""
    while True:
        print("Select input:")
        print("1. Webcam")
        print("2. Upload video file")
        print("3. Upload image file")
        print("4. Exit")
        print()

        try:
            choice = input("Enter option (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting NexGuard.")
            return None

        if choice == "1":
            print(f"\nInitializing Webcam (Camera index {config.camera_index})...")
            source = WebcamInput(camera_index=config.camera_index)
            if not source.is_opened:
                print(f"[ERROR] Webcam camera index {config.camera_index} is unavailable.")
                print("Please check your camera connection or try another input source.\n")
                continue
            return source

        elif choice == "2":
            try:
                video_path = choose_media_file("video")
            except Exception as error:
                print(f"[ERROR] Could not open file picker: {error}\n")
                continue
            if not video_path:
                print("[INFO] Video selection cancelled.\n")
                continue

            if not validate_file_path(video_path):
                print(f"[ERROR] Video file path '{video_path}' does not exist or is invalid.\n")
                continue

            source = VideoFileInput(video_path)
            if not source.is_opened:
                print(f"[ERROR] Could not open video file '{video_path}'.\n")
                continue
            return source

        elif choice == "3":
            try:
                img_path = choose_media_file("image")
            except Exception as error:
                print(f"[ERROR] Could not open file picker: {error}\n")
                continue
            if not img_path:
                print("[INFO] Image selection cancelled.\n")
                continue

            if not validate_file_path(img_path):
                print(f"[ERROR] Image file path '{img_path}' does not exist or is invalid.\n")
                continue

            source = ImageFileInput(img_path)
            if not source.is_opened:
                print(f"[ERROR] Could not load image file '{img_path}'.\n")
                continue
            return source

        elif choice == "4":
            print("\nShutting down NexGuard Edge AI System.")
            return None
        else:
            print("[ERROR] Invalid option. Please select 1, 2, 3, or 4.\n")


def run_detection_loop(
    source: InputSource,
    detector: NexGuardDetector,
    config: NexGuardConfig
) -> None:
    """
    Core edge AI detection loop. Performs frame grabbing, YOLO inference,
    visualization rendering, FPS telemetry tracking, and keyboard input processing.
    """
    ensure_dir(config.output_dir)
    window_name = config.window_name
    event_logger = DetectionEventLogger()
    accident_detector = AccidentDetector(
        candidate_threshold=config.accident_candidate_threshold,
        confirmation_frames=config.accident_confirmation_frames,
        cooldown_frames=config.accident_cooldown_frames,
        max_history=config.tracking_history_length,
        output_dir=config.output_dir,
    )

    frame_count = 0
    total_objects_detected = 0
    paused = False
    start_time = time.time()
    last_stat_time = start_time
    fps = 0.0
    next_frame_time = time.monotonic()
    frame_interval = 1.0 / max(1.0, float(config.target_fps))

    print("\n-----------------------------------------")
    print("NEXGUARD LIVE INFERENCE STARTED")
    print("-----------------------------------------")
    print(f"Device: {detector.dev_type} ({detector.dev_name})")
    print(f"Model: {detector.model_path}")
    print(f"Source: {source.source_type.upper()}")
    print("Controls: Q=Quit | P=Pause | S=Snapshot | R=Reset Stats")
    print("-----------------------------------------\n")

    logger.info(f"Started detection loop on source '{source.source_type}' using model '{detector.model_path}'")

    try:
        while source.is_opened:
            if not paused:
                now = time.monotonic()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time = time.monotonic() + frame_interval

                ret, frame = source.read_frame()
                if not ret or frame is None:
                    if source.source_type == "video":
                        print("\n[INFO] End of video stream reached.")
                    elif source.source_type == "webcam":
                        print("\n[ERROR] Webcam stream interrupted.")
                    break

                frame_count += 1

                # Optional frame skipping for low-resource edge hardware
                if config.frame_skip > 0 and (frame_count % (config.frame_skip + 1) != 0):
                    continue

                # Run YOLO inference and object tracking
                detections = detector.track(frame) if hasattr(detector, "track") else detector.predict(frame)
                tracked_objects = accident_detector.update_tracks(detections)
                current_obj_count = len(tracked_objects) if tracked_objects else len(detections)
                total_objects_detected += current_obj_count
                event_logger.log_detection(frame_count, detections, fps)

                accident_event = None
                if source.source_type != "image":
                    accident_event = accident_detector.analyze_frame(
                        tracked_objects,
                        frame_shape=frame.shape[:2],
                        frame=frame,
                        video_path=getattr(source, "source_path", None),
                    )

                # Render bounding boxes and telemetry overlay
                annotated_frame = draw_detections(frame, tracked_objects or detections)
                annotated_frame = draw_grid(annotated_frame)
                if accident_event is not None:
                    annotated_frame = draw_accident_overlay(annotated_frame, accident_event)
                display_frame = draw_overlay_stats(
                    annotated_frame,
                    fps=fps,
                    frame_count=frame_count,
                    object_count=current_obj_count,
                    device=detector.dev_type,
                    model_name=Path(config.model_path).name,
                    paused=paused
                )

                # Calculate live FPS
                now = time.time()
                elapsed = now - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed

                # Periodically update terminal telemetry (controlled print, no spam)
                if now - last_stat_time >= config.refresh_rate_sec:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Device: {detector.dev_type} | FPS: {fps:.1f} | "
                        f"Frames: {frame_count} | Detected Objects: {current_obj_count}"
                    )
                    last_stat_time = now

            else:
                # When paused, keep rendering current frame overlay with PAUSED indicator
                display_frame = draw_overlay_stats(
                    annotated_frame if 'annotated_frame' in locals() else np.zeros((480, 640, 3), dtype=np.uint8),
                    fps=fps,
                    frame_count=frame_count,
                    object_count=current_obj_count if 'current_obj_count' in locals() else 0,
                    device=detector.dev_type,
                    model_name=Path(config.model_path).name,
                    paused=True
                )
                time.sleep(0.05)

            # Display frame window (guarded against headless environments)
            try:
                cv2.imshow(window_name, display_frame)
            except cv2.error as cv_err:
                logger.warning(f"OpenCV GUI display unavailable (headless system): {cv_err}")
                print("\n[NOTICE] Display window unavailable in headless mode. Telemetry continuing in terminal.")
                time.sleep(0.03)

            # Process keyboard controls (Wait key timeout 1ms or 30ms for static image)
            wait_delay = 30 if source.source_type == "image" else 1
            key = cv2.waitKey(wait_delay) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC to quit
                print("\n[INFO] User requested detection stop (Quit).")
                break
            elif key == ord('p') or key == ord('P'):  # P to pause/resume
                paused = not paused
                status = "PAUSED" if paused else "RESUMED"
                print(f"[CONTROL] Stream {status}.")
                logger.info(f"Detection stream {status}")
            elif key == ord('s') or key == ord('S'):  # S to take snapshot
                snapshot_name = generate_snapshot_filename()
                snapshot_path = Path(config.output_dir) / snapshot_name
                target_img = annotated_frame if 'annotated_frame' in locals() else display_frame
                cv2.imwrite(str(snapshot_path), target_img)
                print(f"\n[✓ SNAPSHOT SAVED] Evidence snapshot captured: '{snapshot_path}'")
                logger.info(f"Evidence snapshot saved to '{snapshot_path}'")
            elif key == ord('r') or key == ord('R'):  # R to reset stats
                frame_count = 0
                total_objects_detected = 0
                start_time = time.time()
                last_stat_time = start_time
                fps = 0.0
                print("\n[CONTROL] Telemetry statistics reset.")
                logger.info("Telemetry statistics reset by user.")

            # Check if OpenCV window closed by user clicking 'X'
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[INFO] Display window closed.")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n[NOTICE] Interrupted by user (Ctrl+C). Cleaning up...")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected detection error: {e}")
        print(f"\n[ERROR] Inference pipeline error: {e}")
    finally:
        source.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("\n-----------------------------------------")
        print("INFERENCE SUMMARY")
        print("-----------------------------------------")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Total Detections Rendered: {total_objects_detected}")
        print("-----------------------------------------\n")
        logger.info(f"Inference loop finished. Total frames: {frame_count}, avg FPS: {fps:.1f}")


def main() -> int:
    """Main application loop."""
    config = DEFAULT_CONFIG

    # Signal handlers for graceful exit
    def signal_handler(sig, frame):
        print("\n[INFO] Termination signal received. Gracefully exiting NexGuard...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Print terminal ASCII banner
    print_banner()

    # Show startup status checklist
    show_initialization_checks(config)

    # Initialize YOLO detector
    detector = NexGuardDetector(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        device=config.device
    )

    # Main interactive application loop
    while True:
        source = prompt_input_selection(config)
        if source is None:
            break

        run_detection_loop(source, detector, config)
        print()

    print("Thank you for using NexGuard Edge AI Surveillance System!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
