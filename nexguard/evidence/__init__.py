"""
NexGuard Evidence Package
=========================
Manages evidentiary snapshots, forensic metadata records, and storage lifecycle.
"""

from nexguard.evidence.collector import EvidenceCollector, IncidentEvidenceRecord

__all__ = ["EvidenceCollector", "IncidentEvidenceRecord"]
