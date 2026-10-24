"""
NexGuard — Detection Pipeline Diagnostic Runner
================================================
Runs standalone diagnostic inference on a single image or video clip.
Annotates detections with bounding boxes, class names, confidence scores,
and inference times, saving output to evidence/diagnostics/.

Usage:
    runner = DiagnosticRunner()
    result = runner.run_image("test_clips/sample.jpg")
    result = runner.run_video("test_clips/traffic.mp4", max_frames=60)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import cfg, MAX_FPS
from nexguard.detection.detector import ObjectDetector
from nexguard.models.schemas import Detection
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.detection.diagnostic")


class DiagnosticRunner:
    """Runs standalone diagnostic inference on image or video files."""

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.detector = detector or ObjectDetector()
        self.output_dir = Path(output_dir or (Path(cfg.evidence_dir).parent / "diagnostics"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_image(
        self,
        image_path: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs diagnostic inference on a single image file.

        Args:
            image_path: Path to input image (.jpg, .png, etc.)
            output_name: Optional custom output filename.

        Returns:
            Dictionary with diagnostics and detection logs.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        t0 = time.monotonic()
        detections = self.detector.detect(frame)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        annotated = self._annotate_frame(frame, detections, frame_idx=1, inference_ms=elapsed_ms)

        out_name = output_name or f"diagnostic_{path.stem}.jpg"
        out_path = self.output_dir / out_name
        cv2.imwrite(str(out_path), annotated)

        # Log detailed detection records
        det_logs = []
        for d in detections:
            log_line = (
                f"Frame: 1 | Class: {d.class_name} (ID={d.class_id}) | "
                f"Conf: {d.confidence:.2f} | "
                f"Box: [{d.bbox[0]:.1f}, {d.bbox[1]:.1f}, {d.bbox[2]:.1f}, {d.bbox[3]:.1f}] | "
                f"Inference: {elapsed_ms:.1f}ms"
            )
            log.info(f"[DIAGNOSTIC] {log_line}")
            det_logs.append({
                "class_name": d.class_name,
                "class_id": d.class_id,
                "confidence": d.confidence,
                "bbox": d.bbox,
                "is_accident": d.is_accident,
            })

        return {
            "source": str(path),
            "output_path": str(out_path),
            "frame_count": 1,
            "detection_count": len(detections),
            "inference_time_ms": elapsed_ms,
            "detections": det_logs,
        }

    def run_video(
        self,
        video_path: str,
        max_frames: int = 90,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs diagnostic inference on a video file.

        Args:
            video_path: Path to video file.
            max_frames: Maximum frames to process in diagnostic run.
            output_name: Optional custom output filename.

        Returns:
            Summary dictionary with performance metrics and detection counts.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_name = output_name or f"diagnostic_{path.stem}.avi"
        out_path = self.output_dir / out_name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, min(source_fps, 15.0), (w, h))

        frame_idx = 0
        total_detections = 0
        class_counts: Dict[str, int] = {}
        inference_times: List[float] = []

        try:
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                t0 = time.monotonic()
                detections = self.detector.detect(frame)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                inference_times.append(elapsed_ms)

                total_detections += len(detections)
                for d in detections:
                    class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                    log.debug(
                        f"[DIAGNOSTIC] Frame {frame_idx:04d} | Class: {d.class_name} | "
                        f"Conf: {d.confidence:.2f} | Time: {elapsed_ms:.1f}ms"
                    )

                annotated = self._annotate_frame(frame, detections, frame_idx, elapsed_ms)
                writer.write(annotated)
        finally:
            cap.release()
            writer.release()

        avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0
        effective_fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0.0

        log.info(
            f"[DIAGNOSTIC COMPLETE] Processed {frame_idx} frames. "
            f"Detections: {total_detections} | Avg Inference: {avg_inference_ms:.1f}ms "
            f"({effective_fps:.1f} FPS cap) | Saved to: {out_path}"
        )

        return {
            "source": str(path),
            "output_path": str(out_path),
            "frame_count": frame_idx,
            "total_detections": total_detections,
            "class_counts": class_counts,
            "avg_inference_ms": avg_inference_ms,
            "effective_fps": min(effective_fps, float(MAX_FPS)),
        }

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int,
        inference_ms: float,
    ) -> np.ndarray:
        """Draws bounding boxes, labels, and diagnostics banner onto frame."""
        annotated = frame.copy()

        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color = (0, 0, 255) if d.is_accident else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{d.class_name} {d.confidence:.0%}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Diagnostic HUD at top-left
        hud = f"Frame: {frame_idx} | Detections: {len(detections)} | Inf: {inference_ms:.1f}ms"
        cv2.putText(
            annotated,
            hud,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        return annotated
