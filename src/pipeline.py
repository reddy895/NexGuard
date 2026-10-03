"""
NexGuard Core Video & Image Processing Pipeline Module
Orchestrates object detection, vehicle tracking, motion analysis, collision candidate evaluation,
temporal accident confirmation, severity scoring, incident recording, HUD rendering, and WhatsApp alerting.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
import cv2
import numpy as np

from config import config
from utils.logger import logger
from utils.video import resize_frame
from utils.validation import (
    validate_image_path,
    validate_video_path,
    validate_webcam_index,
    validate_rtsp_url
)
from src.detector import YOLOObjectDetector
from src.tracker import ObjectTracker
from src.accident_detector import TemporalAccidentDetector, AccidentState
from src.severity_engine import SeverityEngine, SeverityEvaluation
from src.incident_manager import IncidentManager, IncidentRecord
from src.performance import PerformanceMonitor
from src.display import DisplayRenderer
from whatsapp.client import whatsapp_client


class NexGuardPipeline:
    """Unified application pipeline for image, video, webcam, and CCTV RTSP accident detection."""

    def __init__(self, detector: Optional[YOLOObjectDetector] = None):
        self.detector = detector or YOLOObjectDetector()
        self.tracker = ObjectTracker(
            max_lost_frames=config.max_lost_frames,
            iou_threshold=config.iou_threshold
        )
        self.accident_detector = TemporalAccidentDetector(
            confirmation_frames=config.accident_confirmation_frames,
            candidate_threshold=config.accident_candidate_threshold
        )
        self.incident_manager = IncidentManager(cooldown_seconds=config.accident_cooldown_seconds)
        self.performance_monitor = PerformanceMonitor(mode=config.performance_mode)
        self.display = DisplayRenderer()

    def process_image(self, image_path_str: str) -> bool:
        """Processes a single image file."""
        is_valid, err_msg, path_obj = validate_image_path(image_path_str)
        if not is_valid:
            print(f"\nNEXGUARD ERROR: {err_msg}\n")
            return False

        frame = cv2.imread(str(path_obj))
        if frame is None:
            print(f"\nNEXGUARD ERROR: Unable to decode image: {path_obj}\n")
            return False

        # Reset analysis state for image
        self.tracker.reset()
        self.accident_detector.reset()

        detections = self.detector.detect(frame)
        tracked_objects = self.tracker.update(detections)
        
        has_custom_accident = any(d.class_name == "accident" for d in detections)
        analysis = self.accident_detector.process_frame(tracked_objects, custom_model_accident_detected=has_custom_accident)

        severity = SeverityEngine.classify(
            collision_detected=analysis.is_accident,
            collision_score=analysis.confidence_score,
            vehicles_involved=len(analysis.involved_track_ids),
            people_involved=analysis.people_near_accident
        )

        wa_status = whatsapp_client.check_status()["status"]
        incident_id = None

        if analysis.is_accident:
            inc = self.incident_manager.record_incident(
                frame=frame,
                severity_level=severity.level,
                confidence_score=analysis.confidence_score,
                involved_track_ids=analysis.involved_track_ids,
                people_near=analysis.people_near_accident,
                reasons=analysis.reasons,
                source_type="Image"
            )
            if inc:
                incident_id = inc.incident_id
                if whatsapp_client.cached_status.get("connected", False):
                    whatsapp_client.send_accident_alert(
                        incident_id=inc.incident_id,
                        severity_level=severity.level,
                        confidence=analysis.confidence_score,
                        detected_summary=f"Vehicles: {len(analysis.involved_track_ids)}, People: {analysis.people_near_accident}",
                        timestamp=inc.timestamp,
                        source="Image"
                    )

        annotated = self.display.draw_hud(
            frame=frame,
            detections=detections,
            tracked_objects=tracked_objects,
            analysis=analysis,
            severity=severity,
            fps=0.0,
            model_name=self.detector.model_name,
            whatsapp_status=wa_status,
            incident_id=incident_id
        )

        print("\n--------------------------------------------------")
        print("IMAGE DETECTION COMPLETED")
        print(f"Total Objects Detected : {len(detections)}")
        print(f"Tracked Objects        : {len(tracked_objects)}")
        print(f"Accident Status        : {'DETECTED' if analysis.is_accident else 'NORMAL'}")
        print(f"AI Incident Severity   : {severity.level}")
        print("Press 'Q' on the OpenCV window to close, or 'S' to save snapshot.")
        print("--------------------------------------------------\n")

        self.display.init_window()
        cv2.imshow(self.display.window_name, annotated)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('s'), ord('S')]:
                save_p = config.INCIDENTS_DIR / f"manual_image_{int(time.time())}.jpg"
                cv2.imwrite(str(save_p), annotated)
                print(f"[+] Snapshot saved to: {save_p}")

        cv2.destroyAllWindows()
        return True

    def process_video_stream(self, source: Union[str, int], source_name: str = "Video") -> bool:
        """Processes video file, webcam feed, or RTSP stream gracefully."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            if isinstance(source, str) and "rtsp://" in source.lower():
                print(f"\nNEXGUARD ERROR: Unable to connect to CCTV RTSP stream: {source}\n")
            elif source == 0 or isinstance(source, int):
                print(f"\nNEXGUARD ERROR: Unable to access webcam device at index {source}.\n")
            else:
                print(f"\nNEXGUARD ERROR: Unable to open video file: {source}\n")
            return False

        print("\n========================================")
        print("NEXGUARD PERFORMANCE SUMMARY")
        print("========================================")
        print(f"Device         : {self.detector.device.upper()}")
        print(f"Model          : {self.detector.model_name}")
        print(f"Input Source   : {source_name}")
        print(f"Inference Size : {config.inference_size}x{config.inference_size}")
        print(f"Frame Skip     : Every {config.process_every_n_frames} frame(s)")
        print(f"Display Window : {config.display_width}x{config.display_height}")
        print("========================================\n")
        print(f"[+] Processing {source_name}. Controls: Q=Quit, P=Pause/Resume, R=Reset, S=Save Frame\n")

        self.tracker.reset()
        self.accident_detector.reset()
        self.performance_monitor.reset()
        self.display.init_window()

        wa_status = whatsapp_client.check_status()["status"]
        is_paused = False
        latest_annotated = None

        try:
            while True:
                if not is_paused:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"\n[+] End of video stream reached: {source_name}\n")
                        break

                    # Resize frame for uniform display
                    frame = resize_frame(frame, config.display_width, config.display_height)

                    if self.performance_monitor.should_process_frame():
                        detections = self.detector.detect(frame)
                        tracked_objects = self.tracker.update(detections)

                        has_custom_accident = any(d.class_name == "accident" for d in detections)
                        analysis = self.accident_detector.process_frame(tracked_objects, custom_model_accident_detected=has_custom_accident)

                        severity = SeverityEngine.classify(
                            collision_detected=analysis.is_accident,
                            collision_score=analysis.confidence_score,
                            vehicles_involved=len(analysis.involved_track_ids),
                            people_involved=analysis.people_near_accident
                        )

                        self.performance_monitor.tick_processed()
                        fps = self.performance_monitor.get_fps()

                        incident_id = None
                        if analysis.is_accident:
                            inc = self.incident_manager.record_incident(
                                frame=frame,
                                severity_level=severity.level,
                                confidence_score=analysis.confidence_score,
                                involved_track_ids=analysis.involved_track_ids,
                                people_near=analysis.people_near_accident,
                                reasons=analysis.reasons,
                                source_type=source_name
                            )
                            if inc:
                                incident_id = inc.incident_id
                                if whatsapp_client.cached_status.get("connected", False):
                                    whatsapp_client.send_accident_alert(
                                        incident_id=inc.incident_id,
                                        severity_level=severity.level,
                                        confidence=analysis.confidence_score,
                                        detected_summary=f"Vehicles: {len(analysis.involved_track_ids)}, People: {analysis.people_near_accident}",
                                        timestamp=inc.timestamp,
                                        source=source_name
                                    )

                        latest_annotated = self.display.draw_hud(
                            frame=frame,
                            detections=detections,
                            tracked_objects=tracked_objects,
                            analysis=analysis,
                            severity=severity,
                            fps=fps,
                            model_name=self.detector.model_name,
                            whatsapp_status=wa_status,
                            incident_id=incident_id
                        )

                    if latest_annotated is not None:
                        cv2.imshow(self.display.window_name, latest_annotated)

                key = cv2.waitKey(1 if not is_paused else 30) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n[+] User requested quit.\n")
                    break
                elif key in [ord('p'), ord('P')]:
                    is_paused = not is_paused
                    print("[+] Video PAUSED" if is_paused else "[+] Video RESUMED")
                elif key in [ord('r'), ord('R')]:
                    self.accident_detector.reset()
                    self.tracker.reset()
                    self.incident_manager.reset_cooldown()
                    print("[+] Accident analysis state reset.")
                elif key in [ord('s'), ord('S')]:
                    if latest_annotated is not None:
                        save_p = config.INCIDENTS_DIR / f"snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(str(save_p), latest_annotated)
                        print(f"[+] Frame snapshot saved to: {save_p}")
        except KeyboardInterrupt:
            print("\n[+] Graceful shutdown triggered by Ctrl+C.\n")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return True
