"""
NexGuard Geometry & Spatial Math Utilities
"""

import math
from typing import List, Tuple


def get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Returns (center_x, center_y) for bounding box [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def get_bbox_dimensions(bbox: List[float]) -> Tuple[float, float]:
    """Returns (width, height) for bounding box [x1, y1, x2, y2]."""
    return (max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1]))


def get_bbox_area(bbox: List[float]) -> float:
    """Returns area for bounding box [x1, y1, x2, y2]."""
    w, h = get_bbox_dimensions(bbox)
    return w * h


def compute_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Computes Euclidean distance between two 2D points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = get_bbox_area(box1)
    area2 = get_bbox_area(box2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_vector_magnitude(v: Tuple[float, float]) -> float:
    """Computes magnitude of 2D velocity/acceleration vector."""
    return math.hypot(v[0], v[1])


def compute_angle_degrees(v: Tuple[float, float]) -> float:
    """Computes directional angle of vector in degrees [0, 360)."""
    if v[0] == 0 and v[1] == 0:
        return 0.0
    angle = math.degrees(math.atan2(v[1], v[0]))
    return angle % 360.0


def compute_angle_change(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """
    Computes absolute smallest angle change in degrees between two 2D velocity vectors.
    Returns float in range [0.0, 180.0].
    """
    mag1 = compute_vector_magnitude(v1)
    mag2 = compute_vector_magnitude(v2)

    if mag1 < 1e-3 or mag2 < 1e-3:
        return 0.0

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))
