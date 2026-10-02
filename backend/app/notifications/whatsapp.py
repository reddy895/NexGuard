"""
NexGuard WhatsApp Notification & Cooldown Management System
"""

import time
import logging
from typing import Dict, Any, Optional
import requests

from backend.app.config import settings

logger = logging.getLogger("NexGuard.WhatsApp")


class WhatsAppNotifier:
    def __init__(self):
        self.last_alert_time: float = 0.0
        self.cooldown_seconds = settings.ALERT_COOLDOWN_SECONDS

    @property
    def is_configured(self) -> bool:
        return settings.whatsapp_configured

    def get_status(self) -> Dict[str, Any]:
        return {
            "configured": self.is_configured,
            "status": "CONFIGURED" if self.is_configured else "NOT CONFIGURED",
            "cooldown_seconds": self.cooldown_seconds,
            "last_alert_time": self.last_alert_time if self.last_alert_time > 0 else None
        }

    def can_send_alert(self) -> bool:
        if not self.is_configured:
            return False
        now = time.time()
        return (now - self.last_alert_time) >= self.cooldown_seconds

    def format_alert_message(
        self,
        incident_id: str,
        severity_level: str,
        confidence: float,
        detected_summary: str,
        timestamp: str
    ) -> str:
        return (
            "⚠️ *NEXGUARD ALERT*\n\n"
            "Potential road accident detected.\n\n"
            f"*Severity:* {severity_level.upper()}\n"
            f"*Detected Objects:* {detected_summary}\n"
            f"*Confidence:* {int(confidence * 100)}%\n"
            f"*Time:* {timestamp}\n"
            f"*Incident ID:* {incident_id}\n\n"
            "This is an AI-generated incident alert.\n"
            "Please verify the incident through CCTV/authorized personnel."
        )

    def send_incident_alert(
        self,
        incident_id: str,
        severity_level: str,
        confidence: float,
        detected_summary: str,
        timestamp: str
    ) -> Dict[str, Any]:
        """Sends WhatsApp notification via Meta Cloud API or custom webhook when configured."""
        if not self.is_configured:
            logger.info("WhatsApp alert skipped: Credentials NOT CONFIGURED")
            return {
                "sent": False,
                "status": "NOT CONFIGURED",
                "message": "WhatsApp credentials missing from environment."
            }

        if not self.can_send_alert():
            remaining = int(self.cooldown_seconds - (time.time() - self.last_alert_time))
            logger.info(f"WhatsApp alert debounced by cooldown ({remaining}s remaining)")
            return {
                "sent": False,
                "status": "COOLDOWN_ACTIVE",
                "message": f"Alert cooldown active ({remaining}s remaining)."
            }

        message_body = self.format_alert_message(
            incident_id=incident_id,
            severity_level=severity_level,
            confidence=confidence,
            detected_summary=detected_summary,
            timestamp=timestamp
        )

        url = f"https://graph.facebook.com/v18.0/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": settings.WHATSAPP_RECIPIENT_NUMBER,
            "type": "text",
            "text": {"body": message_body}
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5.0)
            if response.status_code in [200, 201]:
                self.last_alert_time = time.time()
                logger.info(f"WhatsApp alert successfully sent for incident {incident_id}")
                return {
                    "sent": True,
                    "status": "SENT",
                    "response": response.json()
                }
            else:
                logger.error(f"WhatsApp API HTTP {response.status_code}: {response.text}")
                return {
                    "sent": False,
                    "status": "API_ERROR",
                    "error": response.text
                }
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return {
                "sent": False,
                "status": "ERROR",
                "error": str(e)
            }


whatsapp_notifier = WhatsAppNotifier()
