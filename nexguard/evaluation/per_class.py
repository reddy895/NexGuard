"""
NexGuard — Per-Class Metrics Analyzer
====================================
Computes granular class-level performance metrics:
- Precision, Recall, Specificity, F1-Score
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Generates formatted comparison markdown tables
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ClassMetrics:
    class_name: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def fpr(self) -> float:
        return 1.0 - self.specificity

    @property
    def fnr(self) -> float:
        return 1.0 - self.recall

class PerClassAnalyzer:
    """
    Renders per-class statistical tables.
    """

    @staticmethod
    def render_markdown_table(metrics: List[ClassMetrics]) -> str:
        lines = [
            "| Class | Support | Precision | Recall | Specificity | F1-Score | FPR | FNR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m in metrics:
            lines.append(
                f"| **{m.class_name}** | {m.support} | {m.precision:.2%} | {m.recall:.2%} | "
                f"{m.specificity:.2%} | {m.f1_score:.2%} | {m.fpr:.2%} | {m.fnr:.2%} |"
            )
        return "\n".join(lines)
