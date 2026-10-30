# NexGuard — Object Detection & Classifier Results

**Date:** 2026-10-30
**Target Architecture:** YOLOv8 Nano (`yolov8n.pt` & `yolov8n-cls.pt`)

---

## 1. Object Detection Performance (COCO Traffic Classes)

Evaluated on standard vehicle (`car`, `bus`, `truck`, `motorcycle`, `bicycle`) and `person` categories:

| Metric | Value |
|---|---|
| **Precision** | 64.00% |
| **Recall** | 53.10% |
| **mAP@50** | 0.528 |
| **mAP@50-95** | 0.373 |
| **Inference Latency** | 90.07 ms |
| **Throughput (FPS)** | 11.1 FPS |

---

## 2. Frame-Level Accident Classifier Performance (Held-Out Test Set)

**Test Set Location:** `dataset_processed/test` (142 samples)

| Metric | Result |
|---|---|
| **Accuracy** | 95.07% |
| **Precision (Accident)** | 96.25% |
| **Recall (Accident)** | 95.06% |
| **F1-Score** | 95.65% |
| **Specificity** | 95.08% |
| **True Positives (TP)** | 77 |
| **False Positives (FP)** | 3 |
| **False Negatives (FN)** | 4 |
| **True Negatives (TN)** | 58 |
| **Inference Latency** | 29.05 ms |
| **Throughput (FPS)** | 34.4 FPS |