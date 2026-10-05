#!/usr/bin/env python3
"""
NexGuard — WhatsApp Test Utility
==================================
Verifies WhatsApp configuration by sending a test message.
Starts the Node.js bot and shows QR code if not already authenticated.

Usage:
    python send_test_message.py
    python send_test_message.py --target 9591152862
    python send_test_message.py --target 9591152862 --message "Custom test"
"""

import argparse
import sys
import time

from nexguard.alerts.whatsapp import WhatsAppNotifier
from nexguard.utils.logging import get_logger

log = get_logger("nexguard.test")

DEFAULT_TARGET = "9591152862"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NexGuard WhatsApp Test — Send a test message via QR login"
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Target phone number with country code (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Custom message text (optional)",
    )
    args = parser.parse_args()

    print("\n" + "═" * 62)
    print("  NEXGUARD WHATSAPP TEST UTILITY")
    print("═" * 62)
    print(f"  Target : +{args.target}")
    print("  The Node.js WhatsApp bot will start.")
    print("  If not authenticated, a QR code will appear below.")
    print("  Scan it with WhatsApp on your phone to log in.")
    print("═" * 62 + "\n")

    notifier = WhatsAppNotifier(enabled=True)

    print("\n  Waiting for WhatsApp bot to initialize...")
    print("  (If QR code appears above, scan it now)\n")

    # Wait for bot to be ready (up to 90s for QR scan)
    for i in range(90):
        status = notifier.get_status()
        if status == "READY":
            break
        if status == "WAITING_QR" and i == 0:
            print("  [QR] Waiting for you to scan the QR code...")
        time.sleep(1)

    status = notifier.get_status()
    print(f"\n  WhatsApp status: {status}")

    if status != "READY":
        print(
            "\n  [ERROR] WhatsApp bot is not ready.\n"
            "  Make sure you scanned the QR code and have a working internet connection."
        )
        notifier.shutdown()
        return 1

    # Send test message
    message = args.message or (
        "*NexGuard Test Message* ✅\n"
        "─────────────────────────\n"
        "WhatsApp integration is working correctly.\n"
        "The NexGuard AI surveillance system is ready.\n\n"
        "_NexGuard AI Safety Monitoring_"
    )

    print(f"\n  Sending test message to +{args.target}...")
    success = notifier.send_test(args.target)

    if success:
        print(f"\n  ✓ SUCCESS — Test message sent to +{args.target}")
        print("  Check your WhatsApp for the message.\n")
    else:
        print(f"\n  ✗ FAILED — Could not send to +{args.target}")
        print(
            "  Check that the number is correct and WhatsApp is connected.\n"
        )

    notifier.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
