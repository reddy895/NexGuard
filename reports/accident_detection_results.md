# NexGuard — Temporal Accident Detection Evaluation Report

**Date:** 2026-10-30
**Evaluation Paradigm:** Multi-Frame Temporal Video & Sequence Evaluation

---

## 1. Event-Level Performance Metrics

| Metric | Result | Benchmark Target |
|---|---|---|
| **Precision** | **100.00%** | ≥ 80.0% |
| **Recall** | **66.67%** | ≥ 75.0% |
| **F1-Score** | **80.00%** | ≥ 78.0% |
| **Specificity** | **100.00%** | ≥ 85.0% |
| **False Positive Rate (FPR)** | **0.00%** | ≤ 15.0% |
| **False Negative Rate (FNR)** | **33.33%** | ≤ 25.0% |
| **Mean Event Detection Latency** | **3.5 frames** | ≤ 6.0 frames |

---

## 2. Event Confusion Matrix

| | Predicted: Accident | Predicted: Normal Traffic |
|---|---|---|
| **Actual: Accident** | 2 (TP) | 1 (FN) |
| **Actual: Normal** | 0 (FP) | 3 (TN) |

---

## 3. Multi-Frame Verification Insights

1. **Temporal Persistence Gating:** Single-frame transient overlaps are suppressed by the consecutive confirmation window.
2. **Parallel Lane Flow Rejection:** Vehicles driving in adjacent lanes at constant velocity are actively filtered.
3. **Zero Fabricated Detections:** All event triggers originate from sustained spatial overlap and kinematic velocity drops.