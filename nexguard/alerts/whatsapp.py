"""
NexGuard — WhatsApp Notifier (Node.js QR Bot Client)
=====================================================
Sends messages by calling the local Node.js whatsapp-web.js HTTP server.
The Node.js bot handles QR code authentication and actual message delivery.

The Python pipeline is never blocked — all HTTP calls run in background threads.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests

from nexguard.utils.logging import get_logger

log = get_logger("nexguard.alerts.whatsapp")

BOT_DIR = Path(__file__).resolve().parents[3] / "whatsapp_bot"
BOT_URL = "http://127.0.0.1:3001"
REQUEST_TIMEOUT = 8


class WhatsAppNotifier:
    """
    Python client for the local Node.js whatsapp-web.js bot.

    On first call, starts the Node.js bot subprocess automatically.
    The bot prints a QR code in the terminal for the user to scan.
    After authentication, messages are dispatched via HTTP POST.

    All send() calls are non-blocking (fire-and-forget threads).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._bot_process: Optional[subprocess.Popen] = None
        self._started = False
        self._status = "NOT_STARTED"

        if not self.enabled:
            log.info("[WhatsApp] Disabled — skipping bot startup")
        else:
            self._start_bot()

    # ------------------------------------------------------------------ #
    #  Bot lifecycle                                                       #
    # ------------------------------------------------------------------ #

    def _start_bot(self) -> None:
        """Starts the Node.js WhatsApp bot as a subprocess."""
        if not BOT_DIR.exists():
            log.error(f"[WhatsApp] Bot directory not found: {BOT_DIR}")
            self.enabled = False
            return

        node_modules = BOT_DIR / "node_modules"
        if not node_modules.exists():
            log.warning("[WhatsApp] node_modules not found — run: cd whatsapp_bot && npm install")
            self.enabled = False
            return

        # Check if already running
        if self._is_bot_alive():
            log.info("[WhatsApp] Bot already running on port 3001")
            self._started = True
            return

        log.info("[WhatsApp] Starting Node.js WhatsApp bot...")
        try:
            self._bot_process = subprocess.Popen(
                ["node", "bot.js"],
                cwd=str(BOT_DIR),
                stdout=sys.stdout,   # Bot QR code goes directly to terminal
                stderr=sys.stderr,
            )
            self._started = True
            log.info(f"[WhatsApp] Bot PID: {self._bot_process.pid}")

            # Wait for it to bind to port (max 30s)
            for _ in range(30):
                time.sleep(1)
                if self._is_bot_alive():
                    log.info("[WhatsApp] Bot HTTP server is up")
                    return
            log.warning("[WhatsApp] Bot did not respond within 30s — continuing anyway")
        except FileNotFoundError:
            log.error("[WhatsApp] 'node' not found in PATH. Install Node.js v16+.")
            self.enabled = False
        except Exception as e:
            log.error(f"[WhatsApp] Failed to start bot: {e}")
            self.enabled = False

    def _is_bot_alive(self) -> bool:
        """Returns True if the bot HTTP server is responding."""
        try:
            resp = requests.get(f"{BOT_URL}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self) -> str:
        """Returns the current WhatsApp bot status string."""
        try:
            resp = requests.get(f"{BOT_URL}/status", timeout=2)
            if resp.status_code == 200:
                return resp.json().get("status", "UNKNOWN")
        except Exception:
            pass
        return "OFFLINE"

    # ------------------------------------------------------------------ #
    #  Message dispatch                                                    #
    # ------------------------------------------------------------------ #

    def send(self, phone: str, message: str) -> None:
        """
        Sends a WhatsApp message asynchronously (non-blocking).

        Args:
            phone:   Recipient number with country code, digits only.
                     Example: '9591152862' (India, no +91 prefix — adds automatically)
            message: Message text.
        """
        if not self.enabled:
            log.info(f"[WhatsApp DISABLED] Would send to {phone}: {message[:60]}...")
            return

        thread = threading.Thread(
            target=self._send_sync,
            args=(phone, message),
            daemon=True,
            name=f"wa-{phone[-4:]}",
        )
        thread.start()

    def _send_sync(self, phone: str, message: str) -> bool:
        """Performs the HTTP POST to the local bot server."""
        # Normalize: strip + and spaces, ensure country code
        phone = phone.strip().lstrip("+").replace(" ", "")
        if not phone.startswith("91") and len(phone) == 10:
            phone = "91" + phone  # Default to India (+91) for 10-digit numbers

        try:
            resp = requests.post(
                f"{BOT_URL}/send",
                json={"phone": phone, "message": message},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200 and resp.json().get("ok"):
                log.info(f"[WhatsApp] Message sent to {phone[-6:]}xxxx")
                return True
            else:
                log.warning(f"[WhatsApp] Send failed: {resp.text[:200]}")
                return False
        except requests.Timeout:
            log.warning("[WhatsApp] Send request timed out")
            return False
        except requests.ConnectionError:
            log.warning("[WhatsApp] Bot not reachable — is it running?")
            return False
        except Exception as e:
            log.error(f"[WhatsApp] Unexpected error: {e}")
            return False

    def send_test(self, phone: str) -> bool:
        """Synchronous test send. Returns True on success."""
        if not self.enabled:
            log.warning("[WhatsApp] Disabled — cannot send test message")
            return False
        return self._send_sync(phone, self._build_test_message())

    def _build_test_message(self) -> str:
        from datetime import datetime
        return (
            "*NexGuard System Test*\n"
            "─────────────────────\n"
            f"Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Status : OPERATIONAL\n"
            "AI Pipeline: READY\n"
            "WhatsApp   : CONNECTED\n\n"
            "This is a test message from NexGuard AI Surveillance.\n"
            "No incident has occurred."
        )

    def shutdown(self) -> None:
        """Terminates the Node.js bot subprocess on exit."""
        if self._bot_process and self._bot_process.poll() is None:
            log.info("[WhatsApp] Terminating bot process...")
            self._bot_process.terminate()
            try:
                self._bot_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._bot_process.kill()
