#!/usr/bin/env python3
"""
NexGuard WhatsApp Standalone Test Utility
Sends a test WhatsApp message to verify authentication and message dispatch.
"""

import sys
import argparse
from config import config
from whatsapp_bot.bot_client import whatsapp_bot


def main():
    parser = argparse.ArgumentParser(description="NexGuard WhatsApp Dispatch Test Utility")
    parser.add_argument("--number", type=str, help="Target WhatsApp recipient phone number (with country code, e.g. 919876543210)")
    parser.add_argument("--message", type=str, help="Custom message string to dispatch")
    args = parser.parse_args()

    recipient = args.number or config.whatsapp_recipient
    if not recipient:
        print("\n==================================================")
        print("         NEXGUARD WHATSAPP DISPATCH TEST")
        print("==================================================")
        recipient = input("Enter recipient WhatsApp number with country code (e.g. 919876543210): ").strip()

    if not recipient:
        print("[-] Error: No recipient phone number provided. Exiting.")
        sys.exit(1)

    msg_body = args.message or (
        "🟢 *NexGuard System Test Message*\n\n"
        "This is an automated test from the NexGuard AI CCTV Accident Detection system.\n"
        "Max 15 FPS Engine: ACTIVE\n"
        "WhatsApp Dispatcher: WORKING\n\n"
        "No accident has occurred."
    )

    print("\n--------------------------------------------------")
    print(f"Target Recipient : {recipient}")
    print("Checking WhatsApp session connection status...")
    status_info = whatsapp_bot.check_status(force_refresh=True)
    print(f"WhatsApp Status  : {status_info['status']}")
    print("--------------------------------------------------")

    print(f"Dispatching test message to {recipient}...")
    success = whatsapp_bot.send_message(recipient, msg_body)

    if success:
        print("\n✓ SUCCESS: Test message dispatched successfully via WhatsApp!")
    else:
        print("\n✗ FAILURE: Could not send WhatsApp message. Ensure session is CONNECTED and number is valid.")
        sys.exit(1)


if __name__ == "__main__":
    main()
