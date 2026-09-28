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
    """
    annotated = frame.copy()

    for det in detections:
        box = det.get("box", [0, 0, 0, 0])
        conf = det.get("confidence", 0.0)
        class_id = det.get("class_id", 0)
        class_name = det.get("class_name", "object")

        x1, y1, x2, y2 = map(int, box)
        color = get_class_color(class_id)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        # Prepare label string
        label = f"{class_name} {conf:.2f}"

        # Get text size for background box
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Draw text background pill
        text_bg_y1 = max(0, y1 - text_height - 6)
        text_bg_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, text_bg_y1),
            (x1 + text_width + 8, text_bg_y2),
            color,
            -1
        )

        # Draw text text
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
