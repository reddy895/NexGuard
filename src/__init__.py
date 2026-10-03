"""
NexGuard Core Source Package
"""

from src.detector import YOLOObjectDetector, DetectionObject
from src.tracker import ObjectTracker, TrackedObject
from src.motion_analyzer import MotionAnalyzer, TrackMotionMetrics
from src.collision_analyzer import CollisionAnalyzer, CollisionCandidate
from src.accident_detector import TemporalAccidentDetector, AccidentAnalysisResult, AccidentState
from src.severity_engine import SeverityEngine, SeverityEvaluation
from src.incident_manager import IncidentManager, IncidentRecord
from src.evidence_manager import EvidenceManager
from src.performance import PerformanceMonitor
from src.display import DisplayRenderer
from src.pipeline import NexGuardPipeline
