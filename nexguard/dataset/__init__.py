"""
NexGuard Dataset Management Package
===================================
Provides reproducible validation, parsing, deduplication, splitting, and preprocessing
for the CCTV road accident dataset.
"""

from nexguard.dataset.validator import DatasetValidator, ValidationSummary
from nexguard.dataset.parser import AnnotationParser, DatasetSample, BoundingBoxAnnotation

__all__ = [
    "DatasetValidator",
    "ValidationSummary",
    "AnnotationParser",
    "DatasetSample",
    "BoundingBoxAnnotation",
]
