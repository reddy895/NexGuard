"""
NexGuard — Confidence Calibration Engine
========================================
Calibrates model and heuristic collision confidence to empirical probabilities:
- Temperature scaling and Platt scaling parameterization
- Expected Calibration Error (ECE) and Brier Score calculation
- Prevents over-confident predictions on edge traffic conditions
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class CalibrationMetrics:
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float

class ConfidenceCalibrator:
    """
    Applies temperature scaling to raw confidence scores.
    """

    def __init__(self, temperature: float = 1.25) -> None:
        self.temperature = max(0.1, temperature)

    def calibrate(self, raw_confidence: float) -> float:
        """
        Scales confidence using logistic temperature transformation.
        """
        # Clip to prevent division by zero or log(0)
        c = min(0.9999, max(0.0001, raw_confidence))
        # Logit
        logit = math.log(c / (1.0 - c))
        # Temperature scale
        scaled_logit = logit / self.temperature
        # Sigmoid
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return round(calibrated, 4)

    @staticmethod
    def compute_ece(
        confidences: List[float],
        ground_truth: List[int],
        num_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Computes Expected Calibration Error (ECE) across prediction bins.
        """
        if not confidences or len(confidences) != len(ground_truth):
            return CalibrationMetrics(0.0, 0.0, 0.0)

        confs = np.array(confidences)
        labels = np.array(ground_truth)

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confs > bin_lower) & (confs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = np.mean(labels[in_bin])
                avg_conf_in_bin = np.mean(confs[in_bin])
                err = abs(acc_in_bin - avg_conf_in_bin)
                ece += err * prop_in_bin
                mce = max(mce, err)

        brier = float(np.mean((confs - labels) ** 2))

        return CalibrationMetrics(
            expected_calibration_error=round(float(ece), 4),
            max_calibration_error=round(float(mce), 4),
            brier_score=round(brier, 4),
        )
