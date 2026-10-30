"""
NexGuard Dataset Management Package
===================================
Provides reproducible validation, parsing, deduplication, splitting, and preprocessing
for the CCTV road accident dataset.
"""

from nexguard.dataset.validator import DatasetValidator, ValidationSummary

__all__ = ["DatasetValidator", "ValidationSummary"]
