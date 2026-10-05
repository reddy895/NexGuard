"""
NexGuard Python WhatsApp Integration Client
Communicates with whatsapp_bot Node.js listener daemon and dispatches accident alert messages.
"""

import os
import sys
import time
import subprocess
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional

from config import config

WHATSAPP_BOT_DIR = Path(__file__).resolve().parent


class WhatsAppBotClient:
    """Manages WhatsApp connection, listener daemon, and automated alert dispatching."""
    
    def __init__(self):
        self.last_alert_time: float = 0.0
        self.cooldown_seconds: int = config.alert_cooldown_seconds
        self.recipient_number: str = config.whatsapp_recipient
        self.cached_status: Dict[str, Any] = {"status": "NOT CONNECTED", "connected": False}
        self.last_status_check: float = 0.0
        self.daemon_proc: Optional[subprocess.Popen] = None

    def check_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Checks WhatsApp session status."""
        now = time.time()
        if not force_refresh and (now - self.last_status_check) < 10.0:
            return self.cached_status

        node_script = WHATSAPP_BOT_DIR / "index.js"
        if not node_script.exists():
            self.cached_status = {"status": "NOT INSTALLED", "connected": False}
            self.last_status_check = now
            return self.cached_status

        try:
            # Check via HTTP server first if daemon is running
            resp = requests.get(f"http://localhost:{config.whatsapp_server_port}/status", timeout=1.5)
            if resp.status_code == 200:
                self.cached_status = {"status": "CONNECTED", "connected": True}
                self.last_status_check = now
                return self.cached_status
        except Exception:
            pass

        # Subprocess status fallback check
        try:
            proc = subprocess.run(
                ["node", str(node_script), "status"],
                cwd=str(WHATSAPP_BOT_DIR),
                capture_output=True,
                text=True,
                timeout=4
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

    def start_listener_daemon(self):
        """Starts the WhatsApp listener daemon in the background."""
        node_script = WHATSAPP_BOT_DIR / "index.js"
        if self.daemon_proc is None or self.daemon_proc.poll() is not None:
            try:
                self.daemon_proc = subprocess.Popen(
                    ["node", str(node_script), "listen"],
                    cwd=str(WHATSAPP_BOT_DIR),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[+] WhatsApp Listener Bot started (PID: {self.daemon_proc.pid})")
            except Exception as e:
                print(f"[-] Failed to launch WhatsApp listener daemon: {e}")

    def start_qr_setup(self):
        """Launches QR code setup terminal prompt."""
        node_script = WHATSAPP_BOT_DIR / "index.js"
        print("\nLaunching WhatsApp QR setup process...")
        try:
            subprocess.run(
                ["node", str(node_script), "qr"],
                cwd=str(WHATSAPP_BOT_DIR),
                check=False
            )
            self.check_status(force_refresh=True)
        except Exception as e:
            print(f"Error starting WhatsApp setup: {e}")

    def can_send_alert(self) -> bool:
        """Enforces alert cooldown."""
        now = time.time()
        return (now - self.last_alert_time) >= self.cooldown_seconds

    def send_accident_alert(
        self,
        incident_id: str,
        severity_level: str,
        confidence: float,
        detected_summary: str,
        timestamp: str
    ) -> bool:
        """Dispatches an accident alert message to the target recipient."""
        target = self.recipient_number or config.whatsapp_recipient
        if not target:
            print("[-] WhatsApp alert dispatch skipped: WHATSAPP_RECIPIENT is empty.")
            return False

        if not self.can_send_alert():
            print("[-] WhatsApp alert cooldown active.")
            return False

        message = (
            "🚨 *NEXGUARD CCTV ACCIDENT ALERT*\n\n"
            "Potential road traffic accident detected!\n\n"
            f"*Incident ID:* {incident_id}\n"
            f"*Severity Level:* {severity_level.upper()}\n"
            f"*Confidence:* {int(confidence * 100)}%\n"
            f"*Involved Entities:* {detected_summary}\n"
            f"*Timestamp:* {timestamp}\n"
            f"*FPS Mode:* Capped Max 15.0 FPS\n\n"
            "Live snapshot saved locally to incidents."
        )

        success = self.send_message(target, message)
        if success:
            self.last_alert_time = time.time()
            print(f"[+] ACCIDENT ALERT DISPATCHED TO WHATSAPP ({target}) SUCCESSFULLY.")
        return success

    def send_message(self, number: str, text: str) -> bool:
        """Sends a message via local HTTP server or fallback node call."""
        # 1. Attempt HTTP dispatch to listener daemon
        try:
            url = f"http://localhost:{config.whatsapp_server_port}/send"
            resp = requests.post(url, json={"number": number, "message": text}, timeout=3.0)
            if resp.status_code == 200 and resp.json().get("success", False):
                return True
        except Exception:
            pass

        # 2. Fallback direct CLI dispatch
        node_script = WHATSAPP_BOT_DIR / "index.js"
        try:
            proc = subprocess.run(
                ["node", str(node_script), "send", number, text],
                cwd=str(WHATSAPP_BOT_DIR),
                capture_output=True,
                text=True,
                timeout=12
            )
            return proc.returncode == 0
        except Exception as e:
            print(f"[-] Error sending WhatsApp message: {e}")
            return False


whatsapp_bot = WhatsAppBotClient()
