# NexGuard — Source-Aware Data Split Report

**Generated:** 2026-10-30
**Strategy:** Source clip grouping (Seed: `42`)
**Target Ratios:** Train: 70%, Val: 15%, Test: 15%

---

## 1. Split Allocation Summary

| Split | Accident Frames | Non-Accident Frames | Total Frames | Actual % |
|---|---|---|---|---|
| **Train** | 314 | 363 | 677 | 68.5% |
| **Validation** | 67 | 79 | 146 | 14.8% |
| **Test** | 81 | 85 | 166 | 16.8% |
| **TOTAL** | **462** | **527** | **989** | **100.0%** |

---

## 2. Leakage Prevention Verification

- **Zero Video Overlap:** All frames sharing an identical video prefix reside exclusively within one split.
- **Independent Evaluation:** The validation and test splits evaluate generalization across previously unseen camera feeds and traffic locations.
- **Reproducibility:** Seed 42 ensures deterministic assignment across any execution.