"""
NexGuard Local WhatsApp Integration Wrapper
Interacts with local Node.js Baileys / wwebjs client.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from config import config, BASE_DIR
from utils.logger import logger

WHATSAPP_DIR = Path(__file__).resolve().parent


class WhatsAppClient:
    """Manages WhatsApp authentication status and sends automated accident alerts."""

    def __init__(self):
        self.last_alert_time: float = 0.0
        self.cooldown_seconds: int = config.alert_cooldown_seconds
        self.recipient_number: str = config.whatsapp_recipient
        self.cached_status: Dict[str, Any] = {"status": "NOT CONNECTED", "connected": False}
        self.last_status_check: float = 0.0

    def check_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Checks WhatsApp session status with rate-limited caching."""
        now = time.time()
        if not force_refresh and (now - self.last_status_check) < 10.0:
            return self.cached_status

        node_script = WHATSAPP_DIR / "index.js"
        if not node_script.exists():
            self.cached_status = {"status": "NOT INSTALLED", "connected": False}
            self.last_status_check = now
            return self.cached_status

        try:
            proc = subprocess.run(
                ["node", str(node_script), "status"],
                cwd=str(WHATSAPP_DIR),
                capture_output=True,
                text=True,
                timeout=3
            )
            output = proc.stdout.strip()
            if "STATUS: CONNECTED" in output:
                self.cached_status = {"status": "CONNECTED", "connected": True}
            elif "AUTHENTICATING" in output:
                self.cached_status = {"status": "AUTHENTICATING", "connected": False}
            else:
                self.cached_status = {"status": "NOT CONNECTED", "connected": False}
        except Exception:
            self.cached_status = {"status": "NOT CONNECTED", "connected": False}

        self.last_status_check = now
        return self.cached_status

    def start_qr_setup(self):
        node_script = WHATSAPP_DIR / "index.js"
        print("\nLaunching WhatsApp QR setup process...")
        try:
            subprocess.run(
                ["node", str(node_script), "qr"],
                cwd=str(WHATSAPP_DIR),
                check=False
            )
            self.check_status(force_refresh=True)
        except Exception as e:
            print(f"Error starting WhatsApp setup: {e}")

    def can_send_alert(self) -> bool:
        now = time.time()
        return (now - self.last_alert_time) >= self.cooldown_seconds

    def send_test_message(self, recipient: Optional[str] = None) -> bool:
        target = recipient or config.whatsapp_recipient
        if not target:
            print("NEXGUARD ERROR: WhatsApp recipient number not set. Configure WHATSAPP_RECIPIENT in .env")
            return False

        message = (
            "NexGuard WhatsApp integration test.\n\n"
            "No accident has been detected.\n"
            "This message confirms that the NexGuard notification system is connected successfully."
        )

        return self.send_message(target, message)

    def send_accident_alert(
        self,
        incident_id: str,
        severity_level: str,
        confidence: float,
        detected_summary: str,
        timestamp: str,
        source: str = "CCTV"
    ) -> Dict[str, Any]:
        if not self.cached_status.get("connected", False):
            return {
                "sent": False,
                "status": "NOT CONNECTED",
                "reason": "WhatsApp session unavailable."
            }

        if not self.can_send_alert():
            remaining = int(self.cooldown_seconds - (time.time() - self.last_alert_time))
            return {
                "sent": False,
                "status": "COOLDOWN_ACTIVE",
                "reason": f"Alert cooldown active ({remaining}s remaining)"
            }

        target = config.whatsapp_recipient
        if not target:
            return {
                "sent": False,
                "status": "NO_RECIPIENT",
                "reason": "WHATSAPP_RECIPIENT is not configured"
            }

        message = (
            "🚨 *NEXGUARD ACCIDENT ALERT*\n\n"
            "Potential road traffic accident detected.\n\n"
            f"*Incident ID:* {incident_id}\n"
            f"*Severity:* {severity_level.upper()}\n"
            f"*Confidence:* {int(confidence * 100)}%\n"
            f"*Summary:* {detected_summary}\n"
            f"*Time:* {timestamp}\n"
            f"*Source:* {source}\n\n"
            "Evidence snapshot captured locally."
        )

        success = self.send_message(target, message)
        if success:
            self.last_alert_time = time.time()
            return {"sent": True, "status": "SENT"}
        else:
            return {"sent": False, "status": "FAILED"}

    def send_message(self, number: str, text: str) -> bool:
        node_script = WHATSAPP_DIR / "index.js"
        try:
            proc = subprocess.run(
                ["node", str(node_script), "send", number, text],
                cwd=str(WHATSAPP_DIR),
                capture_output=True,
                text=True,
                timeout=10
            )
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            return False


whatsapp_client = WhatsAppClient()
