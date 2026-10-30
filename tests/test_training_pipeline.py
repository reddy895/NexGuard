"""
Tests for NexGuard Training Pipeline, Checkpointing, and Device Management
==========================================================================
"""

from pathlib import Path
import pytest
from nexguard.training.device import DeviceManager
from nexguard.training.checkpoint import CheckpointManager
from nexguard.training.model_selector import ModelSelector
from nexguard.training.batch_scaler import MemoryAwareBatchScaler

def test_device_manager_cpu_resolution():
    info = DeviceManager.resolve_device("cpu")
    assert info.device_type == "cpu"
    assert info.device_string == "cpu"
    assert info.amp_supported is False

def test_checkpoint_manager_backup_lifecycle(tmp_path: Path):
    target_dir = tmp_path / "models"
    mgr = CheckpointManager(target_dir)

    dummy_src = tmp_path / "dummy.pt"
    dummy_src.write_bytes(b"MOCK_YOLO_WEIGHTS_BINARY_BYTES" * 100)

    # Save first best
    best_p = mgr.save_best(dummy_src)
    assert best_p.exists()
    assert best_p.read_bytes() == dummy_src.read_bytes()

    # Save new best -> verifies backup of first best
    dummy_src2 = tmp_path / "dummy2.pt"
    dummy_src2.write_bytes(b"UPDATED_WEIGHTS_BINARY_BYTES" * 100)
    best_p2 = mgr.save_best(dummy_src2)
    assert best_p2.read_bytes() == dummy_src2.read_bytes()

    # Verify backup exists
    backups = list(mgr.backup_dir.glob("*.pt"))
    assert len(backups) == 1

def test_model_selector_defaults():
    selection = ModelSelector.select_model(task="classify", force_device="cpu")
    assert "cls" in selection.model_name
    assert selection.task == "classify"
    assert selection.device == "cpu"

def test_batch_scaler_calculation():
    batch = MemoryAwareBatchScaler.calculate_batch_size(
        image_size=224, task="classify", device_override="cpu"
    )
    assert batch in [8, 16]
