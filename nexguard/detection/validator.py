"""
NexGuard — Model and Dataset Validation Engine
===============================================
Provides explicit validation and diagnostic auditing for YOLO weights
and custom accident datasets.

Enforces clear distinctions between:
1. Generic object detectors (COCO pretrained, e.g. yolov8n.pt)
2. Trained accident detection models (fine-tuned on accident datasets)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import cfg, MAX_FPS
from nexguard.models.schemas import ModelDiagnosticInfo, ModelType
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.detection.validator")


def inspect_model_weights(
    model_path: str,
    device: str = "cpu",
    conf: float = 0.25,
    imgsz: int = 640,
) -> ModelDiagnosticInfo:
    """
    Inspects a YOLO weights file without requiring pipeline initialization.

    Identifies:
    - Weight path resolution
    - Model task and class names mapping
    - Generic COCO vs accident-trained classification
    - Dataset availability status
    """
    dataset_p = Path(cfg.accident_dataset_path)
    dataset_status = "FOUND" if dataset_p.exists() else "MISSING (datasets/accident/data.yaml not found)"

    model_p = Path(model_path)
    if not model_p.exists():
        return ModelDiagnosticInfo(
            model_path=str(model_path),
            device=device,
            model_type=ModelType.UNKNOWN,
            class_count=0,
            classes={},
            accident_classes_found=[],
            inference_size=imgsz,
            confidence_threshold=conf,
            max_fps=cfg.max_fps,
            dataset_status=dataset_status,
            dataset_path=str(dataset_p),
            ready=False,
            error_message=f"Model weights file not found: {model_path}",
        )

    try:
        from ultralytics import YOLO
    except ImportError:
        return ModelDiagnosticInfo(
            model_path=str(model_path),
            device=device,
            model_type=ModelType.UNKNOWN,
            class_count=0,
            classes={},
            accident_classes_found=[],
            inference_size=imgsz,
            confidence_threshold=conf,
            max_fps=cfg.max_fps,
            dataset_status=dataset_status,
            dataset_path=str(dataset_p),
            ready=False,
            error_message="ultralytics package not installed",
        )

    try:
        model = YOLO(str(model_p))
        class_names = model.names or {}

        found_accident = [
            name for name in class_names.values()
            if any(acc.lower() in name.lower() for acc in cfg.accident_classes)
        ]
        found_vehicles = [
            name for name in class_names.values()
            if name in cfg.vehicle_classes
        ]

        if found_accident and found_vehicles:
            model_type = ModelType.HYBRID_DETECTOR
        elif found_accident:
            model_type = ModelType.ACCIDENT_DETECTOR
        else:
            model_type = ModelType.GENERIC_DETECTOR

        return ModelDiagnosticInfo(
            model_path=str(model_p.resolve()),
            device=device,
            model_type=model_type,
            class_count=len(class_names),
            classes=dict(class_names),
            accident_classes_found=found_accident,
            inference_size=imgsz,
            confidence_threshold=conf,
            max_fps=cfg.max_fps,
            dataset_status=dataset_status,
            dataset_path=str(dataset_p),
            ready=True,
        )
    except Exception as e:
        return ModelDiagnosticInfo(
            model_path=str(model_p),
            device=device,
            model_type=ModelType.UNKNOWN,
            class_count=0,
            classes={},
            accident_classes_found=[],
            inference_size=imgsz,
            confidence_threshold=conf,
            max_fps=cfg.max_fps,
            dataset_status=dataset_status,
            dataset_path=str(dataset_p),
            ready=False,
            error_message=f"Failed to inspect model: {e}",
        )


def validate_accident_dataset(
    dataset_path: Optional[str] = None,
) -> Tuple[bool, str, Dict[str, any]]:
    """
    Validates the structure of the accident dataset YAML configuration.

    Returns:
        (is_valid, status_message, metadata)
    """
    path_str = dataset_path or cfg.accident_dataset_path
    data_yaml = Path(path_str)

    if not data_yaml.exists():
        msg = (
            f"NEXGUARD ERROR: ACCIDENT DATASET REQUIRED.\n"
            f"Expected dataset YAML at '{path_str}' was not found.\n"
            f"To train or validate accident detection, prepare an annotated dataset. "
            f"See datasets/README.md for instructions."
        )
        return False, msg, {"path": path_str, "status": "MISSING"}

    # Attempt to parse YAML
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        names = data.get("names", {})
        nc = data.get("nc", len(names))

        return True, "Dataset YAML configuration is valid", {
            "path": path_str,
            "status": "VALID",
            "classes": names,
            "num_classes": nc,
            "train": data.get("train"),
            "val": data.get("val"),
        }
    except Exception as e:
        return False, f"Invalid dataset YAML ({e})", {"path": path_str, "status": "CORRUPT"}
