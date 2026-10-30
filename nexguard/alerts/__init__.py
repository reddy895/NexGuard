"""
NexGuard Alerts Package
=======================
Manages alert throttling, duplicate suppression, and notification conditions.
"""

from nexguard.alerts.deduplicator import AlertRateLimiter, AlertDecision

__all__ = ["AlertRateLimiter", "AlertDecision"]
