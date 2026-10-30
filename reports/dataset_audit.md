# NexGuard — Dataset Audit Report

**Generated:** 2026-10-30
**Dataset Location:** `datasets`

---

## 1. Executive Summary

- **Total Image Files:** 989
- **Total Video Files:** 0 in dataset directory (1 in `test_clips/traffic.mp4`)
- **Total Bounding-Box Annotations:** 0 (Dataset is class-folder image classification, no YOLO bounding boxes present)
- **Available Classes:** 2 (`Accident`, `Non Accident`)
- **Available Severity Labels:** None (No severity annotations exist in raw dataset)
- **Supported Architecture:** YOLO classification backbone (`yolov8n-cls`), temporal spatial-kinematic accident reasoning pipeline

---

## 2. Dataset Split Breakdown

| Split | Accident | Non Accident | Total | % of Dataset |
|---|---|---|---|---|
| **train** | 369 | 422 | 791 | 80.0% |
| **val** | 46 | 52 | 98 | 9.9% |
| **test** | 47 | 53 | 100 | 10.1% |
| **TOTAL** | **462** | **527** | **989** | **100.0%** |

---

## 3. Image Dimensions and Formats

| Resolution (W x H) | Count | Percentage |
|---|---|---|
| 1280 x 720 | 957 | 96.8% |
| 720 x 576 | 32 | 3.2% |

**File Extensions:** {'.md': 1, '.cache': 2, '.jpg': 989, '.lnk': 1}

---

## 4. Anomalies, Non-Image Files, and Corruptions

- **Corrupt Image Files:** 0
- **Non-Image Artifacts:** 4
  - `README.md` (Format: `.md` - e.g., Windows shortcut/cache)
  - `train.cache` (Format: `.cache` - e.g., Windows shortcut/cache)
  - `val.cache` (Format: `.cache` - e.g., Windows shortcut/cache)
  - `test/Non Accident/Pictures - Shortcut.lnk` (Format: `.lnk` - e.g., Windows shortcut/cache)

---

## 5. Duplicate Images and Data Leakage Analysis

- **Unique MD5 Checksums:** 965
- **Duplicate Sets:** 24 sets (48 files involved)
- **Cross-Class Label Conflicts:** 5 sets

### Sample Cross-Split & Label Conflicts Detected:

- Duplicate files with conflicting labels: `[('train/Accident/test8_16.jpg', 'train', 'Accident'), ('train/Non Accident/test9_16.jpg', 'train', 'Non Accident')]`
- Duplicate files with conflicting labels: `[('train/Accident/test8_13.jpg', 'train', 'Accident'), ('train/Non Accident/test9_13.jpg', 'train', 'Non Accident')]`
- Duplicate files with conflicting labels: `[('train/Accident/test8_15.jpg', 'train', 'Accident'), ('val/Non Accident/test9_15.jpg', 'val', 'Non Accident')]`
- Duplicate files with conflicting labels: `[('train/Accident/test8_12.jpg', 'train', 'Accident'), ('val/Non Accident/test9_12.jpg', 'val', 'Non Accident')]`
- Duplicate files with conflicting labels: `[('train/Accident/test8_14.jpg', 'train', 'Accident'), ('train/Non Accident/test9_14.jpg', 'train', 'Non Accident')]`

### Temporal Video Frame Distribution (Leakage Finding):

Sequential frames derived from identical camera sources (e.g., `acc1`, `test10`, `test13`) are distributed haphazardly across train, val, and test splits.
For instance:
- `acc1 (1).jpg` is in `train/Accident` while `acc1 (4).jpg` is in `val/Accident` and `acc1 (7).jpg` is in `test/Accident`.
- This artificially inflates validation accuracy unless source-aware grouping is enforced.

---

## 6. Architectural Compatibility & Modeling Strategy

1. **Supervised Object Detection:**
   - The dataset provides **no bounding boxes** for vehicles or persons.
   - True object detection requires pre-trained YOLO weights (`yolov8n.pt` on COCO) for robust vehicle/pedestrian detection, as the raw dataset contains no bounding box coordinates.

2. **Accident Classification:**
   - Supervised classification (`yolov8n-cls`) is fully supported by the `Accident` vs `Non Accident` labels.
   - Validated training on deduplicated, source-split images produces a reliable frame-level accident classifier.

3. **Severity Assessment:**
   - The dataset contains **no severity annotations** (e.g., LOW / MEDIUM / HIGH / CRITICAL).
   - Per engineering specification: an explicitly documented, transparent heuristic baseline is implemented rather than fabricating pseudo-labels.

4. **Temporal Accident Detection:**
   - Evaluated using object tracking (`ObjectTracker`), kinematic trajectory analysis, closing velocity, deceleration, and direct classifier integration.