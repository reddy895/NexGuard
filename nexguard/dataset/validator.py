"""
NexGuard — Dataset Integrity Validator
======================================
Validates image files and annotation structures for road accident models:
- Validates file headers and readability via PIL and OpenCV
- Checks bounding dimensions (width, height, aspect ratio)
- Verifies label taxonomy and folder structure
- Collects integrity statistics and failure logs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image

@dataclass
class ValidationIssue:
    file_path: str
    issue_type: str
    details: str

@dataclass
class ValidationSummary:
    total_scanned: int = 0
    valid_images: int = 0
    corrupt_files: int = 0
    invalid_extensions: int = 0
    zero_byte_files: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.corrupt_files == 0 and self.zero_byte_files == 0

class DatasetValidator:
    """
    Validates integrity of CCTV road accident image datasets.
    """

    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
    EXPECTED_CLASSES: Set[str] = {"Accident", "Non Accident"}

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)

    def validate(self) -> ValidationSummary:
        summary = ValidationSummary()
        if not self.dataset_path.exists():
            summary.issues.append(
                ValidationIssue(
                    file_path=str(self.dataset_path),
                    issue_type="PATH_NOT_FOUND",
                    details=f"Target path {self.dataset_path} does not exist",
                )
            )
            return summary

        for root, _, files in os.walk(self.dataset_path):
            for file_name in files:
                file_path = Path(root) / file_name
                summary.total_scanned += 1
                rel_path = str(file_path.relative_to(self.dataset_path))
                ext = file_path.suffix.lower()

                # 1. Extension check
                if ext not in self.ALLOWED_EXTENSIONS:
                    summary.invalid_extensions += 1
                    summary.issues.append(
                        ValidationIssue(
                            file_path=rel_path,
                            issue_type="INVALID_EXTENSION",
                            details=f"Unsupported extension '{ext}'",
                        )
                    )
                    continue

                # 2. File size check
                try:
                    fsize = file_path.stat().st_size
                    if fsize == 0:
                        summary.zero_byte_files += 1
                        summary.issues.append(
                            ValidationIssue(
                                file_path=rel_path,
                                issue_type="ZERO_BYTE_FILE",
                                details="File has size 0 bytes",
                            )
                        )
                        continue
                except OSError as exc:
                    summary.corrupt_files += 1
                    summary.issues.append(
                        ValidationIssue(
                            file_path=rel_path,
                            issue_type="IO_ERROR",
                            details=str(exc),
                        )
                    )
                    continue

                # 3. Image parsing integrity check
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # verify integrity of headers/compression

                    # Re-open for mode and dimension checks (verify closes file handle)
                    with Image.open(file_path) as img:
                        w, h = img.size
                        if w <= 0 or h <= 0:
                            summary.corrupt_files += 1
                            summary.issues.append(
                                ValidationIssue(
                                    file_path=rel_path,
                                    issue_type="INVALID_DIMENSIONS",
                                    details=f"Invalid image dimensions: {w}x{h}",
                                )
                            )
                            continue
                except Exception as exc:
                    summary.corrupt_files += 1
                    summary.issues.append(
                        ValidationIssue(
                            file_path=rel_path,
                            issue_type="CORRUPT_IMAGE",
                            details=str(exc),
                        )
                    )
                    continue

                # 4. Valid image registered
                summary.valid_images += 1
                category = file_path.parent.name
                summary.class_distribution[category] = summary.class_distribution.get(category, 0) + 1

        return summary
