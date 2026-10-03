"""
Unit Tests for NexGuard Severity Engine
"""

import pytest
from src.severity_engine import SeverityEngine, SeverityEvaluation


def test_severity_normal():
    res = SeverityEngine.classify(collision_detected=False)
    assert res.level == "NORMAL"
    assert res.score == 0.0


def test_severity_low():
    res = SeverityEngine.classify(collision_detected=True, collision_score=0.30, vehicles_involved=1)
    assert res.level == "LOW"


def test_severity_medium():
    res = SeverityEngine.classify(collision_detected=True, collision_score=0.45, vehicles_involved=2, people_involved=0)
    assert res.level == "MEDIUM"


def test_severity_high_pedestrian():
    res = SeverityEngine.classify(collision_detected=True, collision_score=0.55, vehicles_involved=2, people_involved=1)
    assert res.level == "HIGH"


def test_severity_critical():
    res = SeverityEngine.classify(collision_detected=True, collision_score=0.75, vehicles_involved=3, people_involved=2)
    assert res.level == "CRITICAL"
