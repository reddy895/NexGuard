"""
Test Suite: Dataset YAML Loading & Structure Validation
"""

import pytest
import yaml
from pathlib import Path
from utils.config import BASE_DIR

DATA_YAML = BASE_DIR / "training" / "dataset" / "data.yaml"


def test_data_yaml_exists_and_valid():
    assert DATA_YAML.exists()
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    assert "train" in data
    assert "val" in data
    assert "names" in data
    assert data["nc"] == 1
    assert data["names"][0] == "accident"
