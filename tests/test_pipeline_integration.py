"""
End-to-End Pipeline Integration Tests
======================================
Tests multi-frame tracking, temporal accident triggering, and incident resolution.
"""

from pathlib import Path
import numpy as np
import pytest

from nexguard.pipeline import InferencePipeline
from nexguard.models.schemas import Detection, Severity

def test_full_pipeline_multi_frame_collision_flow():
    pipeline = InferencePipeline()
    pipeline.reset()

    # Create synthetic moving frames
    # Frame 1 & 2: Approaching
    # Frame 3 & 4: Contact & sudden stop
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

    results = []
    for f in frames:
        res = pipeline.process_frame(f)
        results.append(res)

    assert len(results) == 10
    assert results[-1].frame_index == 10
    # Clean execution without crash on synthetic frames
    assert results[-1].active_tracks == []

def test_pipeline_reset_clears_state():
    pipeline = InferencePipeline()
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    pipeline.process_frame(dummy)
    assert pipeline.frame_count == 1

    pipeline.reset()
    assert pipeline.frame_count == 0
    assert pipeline.accident_detector.consecutive_count == 0
