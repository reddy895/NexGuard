"""
NexGuard — Early Stopping and Overfitting Monitor
=================================================
Monitors training and validation convergence to prevent overfitting:
- Tracks moving metrics across epochs
- Computes generalization gap (Train Loss vs Validation Loss)
- Halts training after specified patience without performance gains
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.early_stopping")

@dataclass
class EpochMetric:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float

class EarlyStoppingTracker:
    """
    Monitors validation performance and signals when training should terminate.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",  # 'max' for accuracy, 'min' for loss
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_metric: Optional[float] = None
        self.patience_counter: int = 0
        self.should_stop: bool = False
        self.history: List[EpochMetric] = []

    def update(self, epoch: int, train_loss: float, val_loss: float, val_acc: float) -> bool:
        """
        Records metric and returns True if training should halt early.
        """
        metric_val = val_acc if self.mode == "max" else val_loss
        self.history.append(EpochMetric(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_acc
        ))

        # Check for overfitting gap
        gap = val_loss - train_loss
        if gap > 0.5:
            log.warning("Overfitting alert at epoch %d: Val Loss (%.3f) exceeds Train Loss (%.3f) by %.3f", epoch, val_loss, train_loss, gap)

        if self.best_metric is None:
            self.best_metric = metric_val
            self.patience_counter = 0
            return False

        if self.mode == "max":
            improved = metric_val >= (self.best_metric + self.min_delta)
        else:
            improved = metric_val <= (self.best_metric - self.min_delta)

        if improved:
            self.best_metric = metric_val
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                log.info("Early stopping triggered at epoch %d (patience=%d exhausted)", epoch, self.patience)

        return self.should_stop
