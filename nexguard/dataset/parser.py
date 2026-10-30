"""
NexGuard — Dataset Annotation Parser
====================================
Parses and normalizes dataset annotations for both:
1. Classification folder layouts (split/class_name/image.jpg)
2. Bounding-box detection labels (YOLO format: class_id x_center y_center width height)

Features:
- Validates normalized bounding-box bounds [0.0, 1.0]
- Handles coordinate denormalization and format conversion
- Robustly maps class IDs and strings
- Flags missing, malformed, or out-of-bound annotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class BoundingBoxAnnotation:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: Optional[float] = None

    def validate(self) -> List[str]:
        errors = []
        for name, val in [
            ("x_center", self.x_center),
            ("y_center", self.y_center),
            ("width", self.width),
            ("height", self.height),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} out of bounds [0.0, 1.0]: {val}")
        if self.width <= 0:
            errors.append(f"Width must be positive, got {self.width}")
        if self.height <= 0:
            errors.append(f"Height must be positive, got {self.height}")
        return errors

    def to_xyxy(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        xmin = (self.x_center - self.width / 2) * img_width
        ymin = (self.y_center - self.height / 2) * img_height
        xmax = (self.x_center + self.width / 2) * img_width
        ymax = (self.y_center + self.height / 2) * img_height
        return (max(0.0, xmin), max(0.0, ymin), min(float(img_width), xmax), min(float(img_height), ymax))

@dataclass
class DatasetSample:
    image_path: Path
    split: str
    category: str
    class_id: int
    boxes: List[BoundingBoxAnnotation] = field(default_factory=list)
    has_bounding_boxes: bool = False
    source_prefix: str = ""

class AnnotationParser:
    """
    Parses classification and detection annotations from raw dataset directories.
    """

    CLASS_MAPPING: Dict[str, int] = {
        "Non Accident": 0,
        "Accident": 1,
        "non_accident": 0,
        "accident": 1,
    }

    def __init__(self, dataset_root: Path | str) -> None:
        self.dataset_root = Path(dataset_root)

    def parse_sample(self, image_path: Path, label_path: Optional[Path] = None) -> DatasetSample:
        rel = image_path.relative_to(self.dataset_root)
        parts = rel.parts
        split = parts[0] if parts and parts[0] in ["train", "val", "test"] else "unknown"
        category = parts[1] if len(parts) > 1 else image_path.parent.name
        class_id = self.CLASS_MAPPING.get(category, -1)

        # Extract source sequence prefix
        stem = image_path.stem
        prefix = stem.split("_")[0] if "_" in stem else stem.split(" ")[0]

        sample = DatasetSample(
            image_path=image_path,
            split=split,
            category=category,
            class_id=class_id,
            boxes=[],
            has_bounding_boxes=False,
            source_prefix=prefix,
        )

        # Check for YOLO format annotation file
        if label_path and label_path.exists():
            boxes = self.parse_yolo_label_file(label_path)
            sample.boxes = boxes
            sample.has_bounding_boxes = len(boxes) > 0

        return sample

    def parse_yolo_label_file(self, label_path: Path) -> List[BoundingBoxAnnotation]:
        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) < 5:
                    raise ValueError(f"Malformed YOLO annotation at {label_path}:{line_idx}")

                cls_id = int(tokens[0])
                xc, yc, w, h = map(float, tokens[1:5])
                conf = float(tokens[5]) if len(tokens) > 5 else None

                cls_name = "Accident" if cls_id == 1 else "Non Accident"
                box = BoundingBoxAnnotation(
                    class_id=cls_id,
                    class_name=cls_name,
                    x_center=xc,
                    y_center=yc,
                    width=w,
                    height=h,
                    confidence=conf,
                )
                errs = box.validate()
                if errs:
                    raise ValueError(f"Invalid bounding box in {label_path}:{line_idx}: {errs}")
                boxes.append(box)
        return boxes
