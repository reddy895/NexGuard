"""
NexGuard — Severity Classifier Interface
=========================================
Defines the abstract contract for accident severity assessment engines:
- Standardizes input parameters and output schemas across heuristic baselines and trained ML models
- Guarantees transparency: every classification must include auditable evidence reasons
- Prevents unverified medical/casualty claims
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
from nexguard.models.schemas import AccidentEvent, PersonInvolvement, SeverityResult, Track

class SeverityClassifierBase(ABC):
    """
    Abstract interface for accident severity assessment.
    """

    @abstractmethod
    def classify(
        self,
        event: AccidentEvent,
        involved_tracks: List[Track],
        all_tracks: List[Track],
        person_involvements: Optional[List[PersonInvolvement]] = None,
        persistence_frames: int = 0,
    ) -> SeverityResult:
        """
        Calculates severity level and evidence reasons for an accident event.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_trained_model(self) -> bool:
        """
        Returns True if backed by trained neural weights, False if heuristic baseline.
        """
        raise NotImplementedError
