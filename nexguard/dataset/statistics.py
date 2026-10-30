"""
NexGuard — Dataset Statistics and Profiling Engine
==================================================
Computes comprehensive statistical distributions for CCTV accident datasets:
- Class balance, ratio, and imbalance factors
- Channel mean and standard deviation (RGB normalization)
- Aspect ratios and bounding dimensions
- Split proportions and source diversity
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

@dataclass
class ChannelStats:
    mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclass
class DatasetProfile:
    total_images: int = 0
    split_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    class_totals: Dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    resolutions: Dict[str, int] = field(default_factory=dict)
    aspect_ratios: Dict[str, int] = field(default_factory=dict)
    channel_stats: ChannelStats = field(default_factory=ChannelStats)

class DatasetStatistics:
    """
    Computes statistical profiles across dataset directories.
    """

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)

    def profile(self, compute_channel_stats: bool = True, max_samples_for_channels: int = 200) -> DatasetProfile:
        profile = DatasetProfile()
        split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        class_totals: Counter[str] = Counter()
        res_counter: Counter[str] = Counter()
        ar_counter: Counter[str] = Counter()

        all_valid_images: List[Path] = []

        for root, _, files in os.walk(self.dataset_path):
            rel_root = Path(root).relative_to(self.dataset_path)
            parts = rel_root.parts
            split = parts[0] if parts and parts[0] in ["train", "val", "test"] else "all"
            category = parts[1] if len(parts) > 1 else Path(root).name

            for fname in files:
                p = Path(root) / fname
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                all_valid_images.append(p)
                split_counts[split][category] += 1
                class_totals[category] += 1

                try:
                    with Image.open(p) as img:
                        w, h = img.size
                        res_counter[f"{w}x{h}"] += 1
                        ar = round(w / max(h, 1), 2)
                        ar_counter[str(ar)] += 1
                except Exception:
                    pass

        profile.total_images = len(all_valid_images)
        profile.split_counts = {k: dict(v) for k, v in split_counts.items()}
        profile.class_totals = dict(class_totals)
        profile.resolutions = dict(res_counter.most_common(10))
        profile.aspect_ratios = dict(ar_counter.most_common(10))

        # Compute imbalance ratio
        if len(class_totals) >= 2:
            max_c = max(class_totals.values())
            min_c = min(class_totals.values())
            profile.imbalance_ratio = round(max_c / max(min_c, 1), 3)

        # Compute channel stats on a sampled subset for performance
        if compute_channel_stats and all_valid_images:
            sample_step = max(1, len(all_valid_images) // max_samples_for_channels)
            sampled = all_valid_images[::sample_step]
            means = []
            stds = []
            for sp in sampled:
                try:
                    with Image.open(sp) as img:
                        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                        means.append(arr.mean(axis=(0, 1)))
                        stds.append(arr.std(axis=(0, 1)))
                except Exception:
                    pass
            if means:
                overall_mean = np.mean(means, axis=0).round(4).tolist()
                overall_std = np.mean(stds, axis=0).round(4).tolist()
                profile.channel_stats = ChannelStats(mean=overall_mean, std=overall_std)

        return profile

    def export_json(self, output_path: Path) -> None:
        profile = self.profile()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, indent=2)
