"""
NexGuard Display and Visualization Module
Renders bounding boxes, confidence labels, class colors, and live telemetry overlays.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

# Generate distinct RGB colors for class IDs
_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {}


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Generates or retrieves a consistent BGR color for a given class ID."""
    if class_id not in _CLASS_COLORS:
        # Use pseudo-random color mapping based on class_id
        np.random.seed(class_id * 37 + 17)
        color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
        _CLASS_COLORS[class_id] = (color[0], color[1], color[2])  # BGR
    return _CLASS_COLORS[class_id]


def draw_scene_grid(frame: np.ndarray, step: int = 80, color: Tuple[int, int, int] = (60, 60, 60), alpha: float = 0.25) -> np.ndarray:
    """Render a subtle tactical grid over the frame for easier CCTV review."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    overlay = annotated.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)
    cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
    return annotated


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    box_thickness: int = 2,
    font_scale: float = 0.5,
    font_thickness: int = 1
) -> np.ndarray:
    """
    Annotates a video/image frame with bounding boxes, class names, and confidence scores.

    Each detection dictionary contains:
    - 'box': [x1, y1, x2, y2]
    - 'confidence': float (0.0 to 1.0)
    - 'class_id': int
    - 'class_name': str
    - 'track_id': optional int for tracked objects
    """
    annotated = draw_scene_grid(frame)

    for det in detections:
        box = det.get("bbox") or det.get("box") or [0, 0, 0, 0]
        conf = det.get("confidence", 0.0)
        class_id = det.get("class_id", 0)
        class_name = str(det.get("class_name", "object")).lower()
        track_id = det.get("track_id")

        x1, y1, x2, y2 = map(int, box)
        if class_name in {"person", "car", "motorcycle", "bus", "truck", "bicycle"}:
            color = (0, 255, 0)
        else:
            color = (0, 180, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        if track_id is not None:
            label = f"ID: {track_id} {class_name.upper()}"
        else:
            label = f"{class_name.upper()} {conf:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        text_bg_y1 = max(0, y1 - text_height - 6)
        text_bg_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, text_bg_y1),
            (x1 + text_width + 8, text_bg_y2),
            color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return annotated


def draw_accident_overlay(frame: np.ndarray, event: Dict[str, Any]) -> np.ndarray:
    """Render accident candidate and confirmed overlays onto the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    status = str(event.get("status", "suspected")).upper()
    severity = str(event.get("severity", "LOW")).upper()
    confidence = float(event.get("confidence", 0.0)) * 100.0

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 100), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    label = "⚠ ACCIDENT SUSPECTED" if status != "CONFIRMED" else "🚨 ACCIDENT DETECTED"
    color = (0, 165, 255) if status != "CONFIRMED" else (0, 0, 255)
    cv2.putText(annotated, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"SEVERITY: {severity}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"CONFIDENCE: {confidence:.0f}%", (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    collision_box = event.get("collision_box")
    if collision_box:
        x1, y1, x2, y2 = map(int, collision_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(annotated, "ACCIDENT ZONE", (x1 + 8, max(20, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated


def draw_overlay_stats(
    frame: np.ndarray,
    fps: float,
    frame_count: int,
    object_count: int,
    device: str,
    model_name: str,
    paused: bool = False
) -> np.ndarray:
    """
    Renders an edge AI telemetry overlay panel at the top of the video frame.
    Displays live FPS, frame count, detected object count, inference hardware device, and key controls.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Banner overlay background
    overlay_height = 42
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, overlay_height), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Telemetry text
    status_tag = "[PAUSED]" if paused else "[LIVE]"
    text = (
        f"NEXGUARD | {status_tag} | Dev: {device.upper()} | Model: {model_name} | "
        f"FPS: {fps:.1f} | Frames: {frame_count} | Objects: {object_count}"
    )

    cv2.putText(
        annotated,
        text,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 200) if not paused else (0, 165, 255),
        1,
        lineType=cv2.LINE_AA
    )

    # Controls help footer overlay
    controls_text = "Controls: [Q] Quit | [P] Pause | [S] Snapshot | [R] Reset Stats"
    cv2.putText(
        annotated,
        controls_text,
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        lineType=cv2.LINE_AA
    )

    return annotated
