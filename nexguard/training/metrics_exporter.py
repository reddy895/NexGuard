"""
NexGuard — Training Metrics Exporter
===================================
Parses, aggregates, and exports training curves and evaluation histories:
- Ingests training histories and validation metrics
- Exports standardized JSON and Markdown tables
- Calculates convergence rates and performance milestones
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

class MetricsExporter:
    """
    Standardizes training metrics into auditable reports.
    """

    @staticmethod
    def parse_results_csv(csv_path: Path | str) -> List[Dict[str, float]]:
        p = Path(csv_path)
        if not p.exists():
            return []
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {k.strip(): float(v.strip()) for k, v in row.items() if v.strip()}
                rows.append(cleaned)
        return rows

    @staticmethod
    def export_summary_markdown(
        metrics: Dict[str, Any],
        output_path: Path | str,
    ) -> Path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# NexGuard — Model Training Metrics Summary",
            "",
            f"**Model Name:** `{metrics.get('model_name', 'N/A')}`",
            f"**Timestamp:** `{metrics.get('timestamp', 'N/A')}`",
            f"**Device:** `{metrics.get('device', 'N/A')}`",
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Epochs Completed | {metrics.get('epochs_completed', 0)} |",
            f"| Batch Size | {metrics.get('batch_size', 0)} |",
            f"| Image Resolution | {metrics.get('image_size', 0)}x{metrics.get('image_size', 0)} |",
            f"| Validation Top-1 Accuracy | {metrics.get('val_top1_accuracy', 0.0):.2%} |",
            f"| Test Top-1 Accuracy | {metrics.get('test_top1_accuracy', 0.0):.2%} |",
        ]

        p.write_text("\n".join(lines))
        return p
