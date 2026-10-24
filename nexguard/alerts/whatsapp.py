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
from typing import Any, Callable, Dict, List, Optional

import requests

from nexguard.utils.logging import get_logger

log = get_logger("nexguard.alerts.whatsapp")

BOT_DIR = Path(__file__).resolve().parents[2] / "whatsapp_bot"
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
        self._delivery_log: List[Dict[str, Any]] = []
        self.total_sent: int = 0
        self.total_failed: int = 0

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
            resp = requests.get(f"{BOT_URL}/health", timeout=1)
            return resp.status_code == 200
        except Exception:
            return False

    def get_status(self, refresh_sync: bool = False) -> str:
        """
        Returns the current WhatsApp bot status string.
        Uses background caching to ensure inference loop is never blocked.
        """
        now = time.time()
        if not hasattr(self, "_cached_status"):
            self._cached_status = "OFFLINE" if not self.enabled else "INITIALIZING"
            self._last_status_check = 0.0
            self._checking_status = False

        if refresh_sync:
            self._query_status()
            return self._cached_status

        # If cache expired and not already querying, trigger async background poll
        if (now - self._last_status_check) > 3.0 and not getattr(self, "_checking_status", False):
            self._checking_status = True
            threading.Thread(
                target=self._query_status,
                daemon=True,
                name="wa-status-poll"
            ).start()

        return self._cached_status

    def _query_status(self) -> None:
        """Queries the bot status endpoint in background thread."""
        try:
            resp = requests.get(f"{BOT_URL}/status", timeout=1.5)
            if resp.status_code == 200:
                self._cached_status = resp.json().get("status", "UNKNOWN")
            else:
                self._cached_status = "OFFLINE"
        except Exception:
            self._cached_status = "OFFLINE"
        finally:
            self._last_status_check = time.time()
            self._checking_status = False

    @property
    def is_connected(self) -> bool:
        """Returns True only if WhatsApp bot is enabled, reachable, and authenticated."""
        if not self.enabled:
            return False
        status = self.get_status().upper()
        return status in ("READY", "CONNECTED", "AUTHENTICATED", "ONLINE")

    # ------------------------------------------------------------------ #
    #  Message dispatch                                                    #
    # ------------------------------------------------------------------ #

    def send(
        self,
        phone: str,
        message: str,
        on_complete: Optional[Callable[[bool, str], None]] = None,
    ) -> None:
        """
        Sends a WhatsApp message asynchronously (non-blocking).

        Args:
            phone:       Recipient number with country code, digits only.
                         Example: '9591152862' (India, no +91 prefix — adds automatically)
            message:     Message text.
            on_complete: Optional callback invoked with (success, status_str).
        """
        if not self.enabled:
            log.info(f"[WhatsApp DISABLED] Would send to {phone}: {message[:60]}...")
            if on_complete:
                on_complete(False, "DISABLED")
            return

        thread = threading.Thread(
            target=self._send_sync,
            args=(phone, message, on_complete),
            daemon=True,
            name=f"wa-{phone[-4:]}",
        )
        thread.start()

    def _send_sync(
        self,
        phone: str,
        message: str,
        on_complete: Optional[Callable[[bool, str], None]] = None,
    ) -> bool:
        """Performs the HTTP POST to the local bot server."""
        # Normalize: strip + and spaces, ensure country code
        phone = phone.strip().lstrip("+").replace(" ", "")
        if not phone.startswith("91") and len(phone) == 10:
            phone = "91" + phone  # Default to India (+91) for 10-digit numbers

        masked_phone = f"{phone[:4]}****{phone[-4:]}" if len(phone) >= 8 else phone
        start_t = time.time()
        success = False
        status_msg = "UNKNOWN"

        try:
            resp = requests.post(
                f"{BOT_URL}/send",
                json={"phone": phone, "message": message},
                timeout=REQUEST_TIMEOUT,
            )
            elapsed = time.time() - start_t
            if resp.status_code == 200 and resp.json().get("ok"):
                log.info(f"[WhatsApp] Message sent to {masked_phone} ({elapsed:.2f}s)")
                self.total_sent += 1
                success = True
                status_msg = "DELIVERED"
            else:
                err_text = resp.text[:200]
                log.warning(f"[WhatsApp] Send failed to {masked_phone}: {err_text}")
                self.total_failed += 1
                status_msg = f"FAILED: {err_text}"
        except requests.Timeout:
            log.warning(f"[WhatsApp] Send request timed out for {masked_phone}")
            self.total_failed += 1
            status_msg = "TIMEOUT"
        except requests.ConnectionError:
            log.warning(f"[WhatsApp] Bot not reachable for {masked_phone}")
            self.total_failed += 1
            status_msg = "UNREACHABLE"
        except Exception as e:
            log.error(f"[WhatsApp] Unexpected error for {masked_phone}: {e}")
            self.total_failed += 1
            status_msg = f"ERROR: {e}"

        record = {
            "timestamp": time.time(),
            "phone": masked_phone,
            "status": status_msg,
            "success": success,
            "elapsed_s": round(time.time() - start_t, 3),
        }
        self._delivery_log.append(record)

        if on_complete:
            try:
                on_complete(success, status_msg)
            except Exception as cb_err:
                log.error(f"[WhatsApp] on_complete callback error: {cb_err}")

        return success

    def get_delivery_history(self) -> List[Dict[str, Any]]:
        """Returns a copy of all delivery attempt records."""
        return list(self._delivery_log)

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
