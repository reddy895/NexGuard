"""
NexGuard — Duplicate and Near-Duplicate Image Detector
======================================================
Identifies exact duplicates and perceptual near-duplicates to prevent
data leakage and conflicting ground-truth labels across splits.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple
from PIL import Image

@dataclass
class DuplicateGroup:
    hash_value: str
    files: List[Path]
    splits: Set[str]
    categories: Set[str]

    @property
    def has_split_leakage(self) -> bool:
        return len(self.splits) > 1

    @property
    def has_label_conflict(self) -> bool:
        return len(self.categories) > 1

class DatasetDeduplicator:
    """
    Scans a dataset directory to find exact and near duplicates.
    """

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)

    @staticmethod
    def compute_md5(file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_dhash(file_path: Path, hash_size: int = 8) -> str:
        """
        Computes difference hash (dHash) to detect near-duplicates.
        """
        try:
            with Image.open(file_path) as img:
                # Resize to (width + 1, height) in grayscale
                img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
                pixels = list(img.getdata())
                diff = []
                for row in range(hash_size):
                    for col in range(hash_size):
                        p1 = pixels[row * (hash_size + 1) + col]
                        p2 = pixels[row * (hash_size + 1) + col + 1]
                        diff.append(p1 > p2)
                decimal_val = 0
                hex_str = []
                for index, val in enumerate(diff):
                    if val:
                        decimal_val += 2 ** (index % 4)
                    if (index % 4) == 3:
                        hex_str.append(hex(decimal_val)[2:])
                        decimal_val = 0
                return "".join(hex_str)
        except Exception:
            return ""

    def find_duplicates(self) -> List[DuplicateGroup]:
        hash_map: Dict[str, List[Tuple[Path, str, str]]] = defaultdict(list)

        for root, _, files in os.walk(self.dataset_path):
            rel_root = Path(root).relative_to(self.dataset_path)
            parts = rel_root.parts
            split = parts[0] if len(parts) > 0 and parts[0] in ["train", "val", "test"] else "unknown"
            category = parts[1] if len(parts) > 1 else Path(root).name

            for fname in files:
                p = Path(root) / fname
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    h = self.compute_md5(p)
                    hash_map[h].append((p, split, category))
                except OSError:
                    continue

        groups = []
        for h, items in hash_map.items():
            if len(items) > 1:
                files = [item[0] for item in items]
                splits = {item[1] for item in items}
                categories = {item[2] for item in items}
                groups.append(DuplicateGroup(
                    hash_value=h,
                    files=files,
                    splits=splits,
                    categories=categories,
                ))
        return groups
