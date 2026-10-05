"""
NexGuard — Alert Manager
==========================
Orchestrates incident alert routing, severity-based contact selection,
and cooldown enforcement.

The default target number (9591152862) is set in .env.example.
All sends are non-blocking — the AI detection pipeline is never paused.

Alert routing by severity:
    LOW      → log only
    MEDIUM   → configured contacts
    HIGH     → configured contacts
    CRITICAL → all contacts
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from config import cfg
from nexguard.alerts.whatsapp import WhatsAppNotifier
from nexguard.models.schemas import Incident, Severity
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.alerts.manager")

# Default alert target (override in .env: NEXGUARD_WHATSAPP_RECIPIENTS)
DEFAULT_TARGET = "9591152862"


class AlertManager:
    """
    Central alert routing and dispatch system.

    Builds informative alert messages and dispatches them via WhatsApp.
    All sends are asynchronous and never block the detection loop.
    """

    def __init__(self, notifier: Optional[WhatsAppNotifier] = None) -> None:
        self._notifier = notifier or WhatsAppNotifier(
            enabled=cfg.whatsapp_enabled,
        )

    def dispatch(self, incident: Incident) -> bool:
        """
        Dispatches an alert for a confirmed incident.
        Returns True if at least one message was queued for sending.
        """
        # Determine recipients
        contacts = cfg.get_contacts_for_severity(incident.severity.value)
        if not contacts:
            # Fall back to default target if nothing configured
            contacts = cfg.whatsapp_recipients or [DEFAULT_TARGET]

        # LOW severity: log only unless contacts explicitly configured
        if incident.severity == Severity.LOW and not cfg.alert_low_contacts:
            log.info(
                f"[ALERT] LOW severity {incident.incident_id} — log only"
            )
            self._log_alert(incident)
            return False

        message = self._build_message(incident)

        for contact in contacts:
            log.info(
                f"[ALERT] Sending to +{contact} — "
                f"{incident.incident_id} [{incident.severity.value}]"
            )
            self._notifier.send(contact, message)

        self._log_alert(incident)
        return True

    def _build_message(self, incident: Incident) -> str:
        """Builds a concise, structured WhatsApp alert message."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        severity_emoji = {
            "LOW": "🟡",
            "MEDIUM": "🟠",
            "HIGH": "🔴",
            "CRITICAL": "🚨",
        }.get(incident.severity.value, "⚠️")

        lines = [
            f"{severity_emoji} *NEXGUARD INCIDENT ALERT*",
            "─" * 30,
            f"*ID*       : {incident.incident_id}",
            f"*Time*     : {now}",
            f"*Location* : {cfg.location_name}",
            f"*Type*     : {incident.event_type.value.replace('_', ' ').title()}",
            f"*Severity* : {incident.severity.value}",
            f"*Score*    : {incident.severity_score:.0%}",
            f"*Confidence*: {incident.confidence:.0%}",
            "",
        ]

        if incident.involved_vehicles:
            lines.append(f"🚗 Vehicles involved: {len(incident.involved_vehicles)}")

        if incident.involved_people:
            lines.append(f"🚶 Persons at risk: {len(incident.involved_people)}")
            for p in incident.involved_people[:3]:
                lines.append(
                    f"  Person #{p.track_id}: *{p.involvement_level}* risk "
                    f"({p.movement}, {p.distance_px:.0f}px away)"
                )

        if incident.severity_reasons:
            lines.append("")
            lines.append("*Signals detected:*")
            for r in incident.severity_reasons[:4]:
                lines.append(f"  • {r}")

        lines.extend([
            "",
            "⚠️ _PROTOTYPE SYSTEM — Verify before emergency response._",
            "_NexGuard AI Safety Monitoring_",
        ])

        return "\n".join(lines)

    def _log_alert(self, incident: Incident) -> None:
        log.info(
            f"[ALERT LOG] {incident.incident_id} | "
            f"severity={incident.severity.value} | "
            f"score={incident.severity_score:.3f} | "
            f"vehicles={len(incident.involved_vehicles)} | "
            f"people={len(incident.involved_people)}"
        )

    def shutdown(self) -> None:
        """Cleanly shuts down the notifier (terminates Node.js bot)."""
        self._notifier.shutdown()
