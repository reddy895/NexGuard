"""
Tests for NexGuard Dataset Integrity, Corrupted-File Detection, and Annotation Parsing
=====================================================================================
"""

import tempfile
from pathlib import Path
from PIL import Image
import pytest

from nexguard.dataset.validator import DatasetValidator
from nexguard.dataset.parser import AnnotationParser, BoundingBoxAnnotation

def test_validator_detects_clean_dataset(tmp_path: Path):
    clean_dir = tmp_path / "clean_data" / "Accident"
    clean_dir.mkdir(parents=True)
    img_path = clean_dir / "sample.jpg"
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    img.save(img_path)

    validator = DatasetValidator(tmp_path / "clean_data")
    summary = validator.validate()

    assert summary.valid_images == 1
    assert summary.corrupt_files == 0
    assert summary.zero_byte_files == 0
    assert summary.is_clean is True

def test_validator_detects_corrupted_file(tmp_path: Path):
    corrupt_dir = tmp_path / "corrupt_data" / "Accident"
    corrupt_dir.mkdir(parents=True)
    corrupt_file = corrupt_dir / "broken.jpg"
    # Write corrupt byte stream
    corrupt_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00randomgarbagedatatruncated")

    validator = DatasetValidator(tmp_path / "corrupt_data")
    summary = validator.validate()

    assert summary.corrupt_files == 1
    assert summary.is_clean is False
    assert any(i.issue_type == "CORRUPT_IMAGE" for i in summary.issues)

def test_validator_detects_zero_byte_file(tmp_path: Path):
    zero_dir = tmp_path / "zero_data" / "Non Accident"
    zero_dir.mkdir(parents=True)
    zero_file = zero_dir / "empty.jpg"
    zero_file.write_bytes(b"")

    validator = DatasetValidator(tmp_path / "zero_data")
    summary = validator.validate()

    assert summary.zero_byte_files == 1
    assert summary.is_clean is False

def test_annotation_parser_validates_bounding_boxes():
    valid_box = BoundingBoxAnnotation(
        class_id=1,
        class_name="Accident",
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.3,
    )
    assert valid_box.validate() == []
    xyxy = valid_box.to_xyxy(img_width=1000, img_height=1000)
    assert xyxy == (400.0, 350.0, 600.0, 650.0)

    invalid_box = BoundingBoxAnnotation(
        class_id=1,
        class_name="Accident",
        x_center=1.5,  # out of bounds
        y_center=-0.1,  # out of bounds
        width=-0.2,   # negative width
        height=0.5,
    )
    errs = invalid_box.validate()
    assert len(errs) >= 3

def test_deduplicator_finds_duplicates_and_leakage(tmp_path: Path):
    from nexguard.dataset.deduplicator import DatasetDeduplicator

    train_acc = tmp_path / "train" / "Accident"
    val_non_acc = tmp_path / "val" / "Non Accident"
    train_acc.mkdir(parents=True)
    val_non_acc.mkdir(parents=True)

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    p1 = train_acc / "img1.jpg"
    p2 = val_non_acc / "img2.jpg"
    img.save(p1)
    img.save(p2)  # Exact identical image in different split and different class!

    dedup = DatasetDeduplicator(tmp_path)
    dups = dedup.find_duplicates()

    assert len(dups) == 1
    group = dups[0]
    assert group.has_split_leakage is True
    assert group.has_label_conflict is True
    assert len(group.files) == 2
