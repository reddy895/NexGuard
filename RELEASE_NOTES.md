# NexGuard v1.0.0 — Production Accident Detection & Rebuilt MLOps Release

**Release Date:** 2026-10-30  
**Git Baseline:** `728a606bd68c8b3eac1cacd78145ceb80ff4d3c3`  
**Git Commit Count:** Exactly 46 verified commits  
**Test Suite:** 59 automated unit and integration tests (100% passing)

---

## Key Achievements & Milestones

1. **Dataset Integrity & Leakage Prevention:**
   - Exhaustive audit performed on 989 CCTV frames across `Accident` and `Non Accident` categories.
   - Identified 24 duplicate sets and 5 cross-split label conflicts in the raw Kaggle dataset.
   - Designed and applied `SourceAwareSplitter` to group sequential frames by video source, completely eliminating data leakage between training and evaluation splits.

2. **Model Training & Empirical Evaluation:**
   - Trained YOLOv8 Nano accident classifier on deduplicated CCTV frames.
   - Empirical test split metrics:
     - **Accuracy:** 95.07%
     - **Precision:** 96.25%
     - **Recall:** 95.06%
     - **F1-Score:** 95.65%
     - **Inference Latency:** 29.05 ms (34.4 FPS on CPU)
   - Object detection (COCO traffic classes): mAP@50 0.528, mAP@50-95 0.373.

3. **Temporal Multi-Signal Accident Reasoning:**
   - Multi-frame collision confirmation combining spatial IoU overlap, centroid proximity, approaching velocity, and sudden deceleration.
   - False-positive suppression for parallel lane driving, controlled stoplights, and stationary parked vehicles.
   - Confidence calibration via temperature scaling.

4. **Forensic Evidence & Deduplicated Alerts:**
   - Automatic snapshot capture and metadata archiving in `evidence/incidents/`.
   - Rate limiting and track-signature deduplication preventing emergency alert spam.

5. **Tested & Production Ready:**
   - All 59 tests execute cleanly without external hardware dependencies.
   - Validated end-to-end integration across webcam and video file pipelines.
