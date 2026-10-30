"""
NexGuard — CLI Dataset Validation Script
========================================
Usage:
    python scripts/validate_dataset.py --data dataset/
    python scripts/validate_dataset.py --data datasets/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexguard.dataset.validator import DatasetValidator

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate NexGuard CCTV accident dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset",
        help="Path to dataset directory (default: dataset/)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists() and (Path("datasets")).exists():
        data_path = Path("datasets")

    print(f"==================================================")
    print(f"  NEXGUARD DATASET VALIDATION PIPELINE")
    print(f"==================================================")
    print(f"Target Directory: {data_path.resolve()}")

    validator = DatasetValidator(data_path)
    summary = validator.validate()

    print(f"\n--- Validation Summary ---")
    print(f"Total Scanned      : {summary.total_scanned}")
    print(f"Valid Images       : {summary.valid_images}")
    print(f"Corrupt Files      : {summary.corrupt_files}")
    print(f"Invalid Extensions : {summary.invalid_extensions}")
    print(f"Zero Byte Files    : {summary.zero_byte_files}")
    print(f"Class Distribution : {summary.class_distribution}")

    if summary.issues:
        print(f"\nIssues Detected ({len(summary.issues)}):")
        for issue in summary.issues[:10]:
            print(f"  [{issue.issue_type}] {issue.file_path}: {issue.details}")
        if len(summary.issues) > 10:
            print(f"  ... and {len(summary.issues) - 10} more issues.")

    if summary.is_clean:
        print("\n[SUCCESS] Dataset integrity verified. No corrupt or zero-byte files.")
        return 0
    else:
        print("\n[WARNING] Dataset contains corrupt or non-standard files.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
