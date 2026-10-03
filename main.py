"""
NexGuard — Terminal-First AI CCTV Accident Detection System
Main Application Entry Point
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from config import config, BASE_DIR, INCIDENTS_DIR
from utils.logger import logger
from utils.validation import (
    validate_image_path,
    validate_video_path,
    validate_webcam_index,
    validate_rtsp_url
)
from src.detector import YOLOObjectDetector
from src.pipeline import NexGuardPipeline
from src.severity_engine import SeverityEngine
from whatsapp.client import whatsapp_client
from training.train_accident import train_accident_model
from training.validate_accident import validate_accident_model


def print_banner(pipeline: NexGuardPipeline, wa_status: str):
    detector = pipeline.detector
    custom_status = "READY" if detector.custom_model_loaded else "NOT INSTALLED"

    print("\n" + "=" * 50)
    print("                 NEXGUARD")
    print("        AI CCTV ACCIDENT DETECTION")
    print("=" * 50)
    print(f"BASE YOLO MODEL : {'READY' if detector.ready else 'ERROR'} ({detector.model_name})")
    print(f"CUSTOM ACCIDENT : {custom_status}")
    print(f"DEVICE          : {detector.device.upper()}")
    print("TRACKER         : READY")
    print("TEMPORAL ENGINE : READY")
    print(f"WHATSAPP        : {wa_status}")
    print("=" * 50 + "\n")


def print_menu():
    print("========================================")
    print("             NEXGUARD MENU")
    print("========================================")
    print("1. Image Detection")
    print("2. Video Detection")
    print("3. Webcam Detection")
    print("4. RTSP/CCTV Detection")
    print("5. Train Accident Model")
    print("6. Validate Accident Model")
    print("7. Model & System Status")
    print("8. WhatsApp Setup")
    print("9. Severity Tests & Simulation")
    print("10. Exit")
    print("========================================")


def print_system_status(pipeline: NexGuardPipeline):
    detector = pipeline.detector
    wa_info = whatsapp_client.check_status(force_refresh=True)

    print("\n========================================")
    print("        NEXGUARD SYSTEM STATUS")
    print("========================================")
    print(f"Base Model      : {'READY' if detector.ready else 'ERROR'} ({detector.model_name})")
    print(f"Custom Model    : {'READY' if detector.custom_model_loaded else 'NOT INSTALLED'} ({config.custom_model_path})")
    print(f"Device          : {detector.device.upper()}")
    print("Tracker         : READY (Persistent Overlap/Distance Cost)")
    print("Temporal Engine : READY (5-Frame Confirmation State Machine)")
    print("Severity Engine : READY (NORMAL / LOW / MEDIUM / HIGH / CRITICAL)")
    print("Incident Manager: READY (Cooldown & Evidence Persistence)")
    print(f"WhatsApp Status : {wa_info['status']}")
    print(f"Recipient       : {config.whatsapp_recipient or 'NOT CONFIGURED'}")
    print(f"Performance Mode: {config.performance_mode.upper()} (Skip: {config.process_every_n_frames}, ImgSz: {config.inference_size})")
    print("========================================\n")


def run_whatsapp_menu():
    while True:
        print("\n==================================================")
        print("            WHATSAPP SETUP & CONTROLS")
        print("==================================================")
        status_info = whatsapp_client.check_status(force_refresh=True)
        print(f"Current Status: {status_info['status']}")
        print(f"Recipient     : {config.whatsapp_recipient or 'NOT CONFIGURED'}")
        print("--------------------------------------------------")
        print("1. Scan QR Code / Authenticate Session")
        print("2. Send Test WhatsApp Message")
        print("3. Set Recipient Phone Number")
        print("4. Back to Main Menu")
        print("==================================================")

        choice = input("Enter option (1-4): ").strip()
        if choice == "1":
            whatsapp_client.start_qr_setup()
        elif choice == "2":
            print("\nSending test message...")
            success = whatsapp_client.send_test_message()
            if success:
                print("✓ RESULT: SUCCESS — WhatsApp test message sent successfully.")
            else:
                print("✗ RESULT: FAILED — Ensure session is CONNECTED and recipient number is valid.")
        elif choice == "3":
            num = input("Enter recipient number with country code (e.g. 919876543210): ").strip()
            if num:
                config.whatsapp_recipient = num
                whatsapp_client.recipient_number = num
                env_file = BASE_DIR / ".env"
                with open(env_file, "a") as f:
                    f.write(f"\nWHATSAPP_RECIPIENT={num}\n")
                print(f"[+] Recipient updated to {num} and saved to .env")
        elif choice == "4":
            break


def run_severity_test_suite():
    print("\n==================================================")
    print("         SEVERITY AUTOMATED TEST SUITE")
    print("==================================================")

    scenarios = [
        {"test_id": 1, "name": "TEST 1: No collision", "expected": "NORMAL", "params": {"collision_detected": False, "collision_score": 0.10, "vehicles_involved": 0, "people_involved": 0}},
        {"test_id": 2, "name": "TEST 2: Minor vehicle interaction", "expected": "LOW", "params": {"collision_detected": True, "collision_score": 0.30, "vehicles_involved": 1, "people_involved": 0}},
        {"test_id": 3, "name": "TEST 3: Clear collision between vehicles", "expected": "MEDIUM", "params": {"collision_detected": True, "collision_score": 0.40, "vehicles_involved": 2, "people_involved": 0}},
        {"test_id": 4, "name": "TEST 4: Collision + person involvement", "expected": "HIGH", "params": {"collision_detected": True, "collision_score": 0.35, "vehicles_involved": 2, "people_involved": 1}},
        {"test_id": 5, "name": "TEST 5: Strong collision + multi-people/vehicles", "expected": "CRITICAL", "params": {"collision_detected": True, "collision_score": 0.55, "vehicles_involved": 3, "people_involved": 2, "has_vulnerable_user": True, "post_stop": True}},
    ]

    all_passed = True
    for sc in scenarios:
        res = SeverityEngine.classify(**sc["params"])
        passed = (res.level == sc["expected"])
        if not passed:
            all_passed = False
        status_str = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{sc['name']:<50} | Expected: {sc['expected']:<8} | Evaluated: {res.level:<8} | {status_str}")

    print("--------------------------------------------------")
    if all_passed:
        print("ALL 5 SEVERITY TEST SCENARIOS PASSED SUCCESSFULLY.")
    else:
        print("SOME SEVERITY TESTS FAILED.")
    print("==================================================\n")


def main():
    try:
        pipeline = NexGuardPipeline()
        wa_info = whatsapp_client.check_status()
        print_banner(pipeline, wa_info["status"])

        while True:
            print_menu()
            choice = input("\nSelect choice (1-10): ").strip()

            if choice == "1":
                img_path = input("\nEnter image file path: ").strip().strip('"').strip("'")
                if img_path:
                    pipeline.process_image(img_path)
            elif choice == "2":
                vid_path = input("\nEnter video file path: ").strip().strip('"').strip("'")
                if vid_path:
                    pipeline.process_video_stream(vid_path, source_name="Video")
            elif choice == "3":
                print("\n[+] Accessing Webcam (Index 0)...")
                pipeline.process_video_stream(0, source_name="Webcam")
            elif choice == "4":
                rtsp = input("\nEnter CCTV / RTSP URL: ").strip()
                if rtsp:
                    pipeline.process_video_stream(rtsp, source_name="CCTV")
            elif choice == "5":
                epochs_str = input("Enter training epochs (default 50): ").strip()
                epochs = int(epochs_str) if epochs_str.isdigit() else 50
                train_accident_model(epochs=epochs)
                pipeline = NexGuardPipeline()  # Reload pipeline
            elif choice == "6":
                validate_accident_model()
            elif choice == "7":
                print_system_status(pipeline)
            elif choice == "8":
                run_whatsapp_menu()
            elif choice == "9":
                run_severity_test_suite()
            elif choice == "10":
                print("\nExiting NexGuard. Goodbye!\n")
                sys.exit(0)
            else:
                print("\nInvalid selection. Please enter a number between 1 and 10.\n")
    except KeyboardInterrupt:
        print("\n\nExiting NexGuard gracefully. Goodbye!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
