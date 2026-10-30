"""
NexGuard — Dataset Preprocessing Pipeline
=========================================
Builds a clean, reproducible, leak-free training dataset:
1. Validates all raw image files
2. Eliminates exact duplicate files and resolves label conflicts
3. Partitions images via source-aware video clip grouping
4. Materializes a clean directory structure into dataset_processed/
5. Generates a machine-readable manifest (dataset_processed/manifest.json)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set
from PIL import Image

from nexguard.dataset.deduplicator import DatasetDeduplicator
from nexguard.dataset.splitter import SourceAwareSplitter
from nexguard.dataset.validator import DatasetValidator

class DatasetPreprocessor:
    """
    Orchestrates the entire dataset cleaning and preparation pipeline.
    """

    def __init__(
        self,
        raw_dir: Path | str,
        output_dir: Path | str = "dataset_processed",
        seed: int = 42,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed

    def run(self) -> Dict[str, any]:
        print(f"[PREPROCESSOR] Starting preprocessing: {self.raw_dir} -> {self.output_dir}")

        # 1. Validation
        validator = DatasetValidator(self.raw_dir)
        summary = validator.validate()
        print(f"[PREPROCESSOR] Scanned {summary.total_scanned} files. Valid images: {summary.valid_images}")

        # 2. Duplicate detection
        deduplicator = DatasetDeduplicator(self.raw_dir)
        duplicate_groups = deduplicator.find_duplicates()
        conflicted_hashes: Set[str] = {g.hash_value for g in duplicate_groups if g.has_label_conflict}
        print(f"[PREPROCESSOR] Found {len(duplicate_groups)} duplicate groups ({len(conflicted_hashes)} label conflicts)")

        # 3. Source-aware splitting
        splitter = SourceAwareSplitter(self.raw_dir, seed=self.seed)
        split_plan = splitter.split()

        # 4. Materialize dataset_processed/
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "source_dir": str(self.raw_dir),
            "output_dir": str(self.output_dir),
            "splits": {"train": {}, "val": {}, "test": {}},
            "total_processed": 0,
            "conflicts_excluded": len(conflicted_hashes),
        }

        seen_hashes: Set[str] = set()

        for split_name, cat_dict in split_plan.items():
            for category, file_list in cat_dict.items():
                target_cat_dir = self.output_dir / split_name / category
                target_cat_dir.mkdir(parents=True, exist_ok=True)

                copied_count = 0
                for fpath in file_list:
                    h = DatasetDeduplicator.compute_md5(fpath)
                    # Discard conflicting duplicates to preserve data purity
                    if h in conflicted_hashes or h in seen_hashes:
                        continue

                    seen_hashes.add(h)
                    dest_file = target_cat_dir / fpath.name
                    # Handle name collision within destination folder
                    if dest_file.exists():
                        dest_file = target_cat_dir / f"{fpath.stem}_{h[:6]}{fpath.suffix}"

                    shutil.copy2(fpath, dest_file)
                    copied_count += 1

                manifest["splits"][split_name][category] = copied_count
                manifest["total_processed"] += copied_count

        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[PREPROCESSOR] Complete. Materialized {manifest['total_processed']} images into {self.output_dir}")
        print(f"[PREPROCESSOR] Manifest saved to: {manifest_path}")
        return manifest
