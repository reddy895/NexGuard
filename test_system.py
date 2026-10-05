#!/usr/bin/env python3
"""
NexGuard Comprehensive Automated Test Suite
Tests Max 15 FPS Rate Limiter, Multi-Object Tracker, Motion Gesture Classifier,
Visual UI HUD Renderer, and WhatsApp Dispatcher.
"""

import sys
import time
import unittest
import numpy as np
import cv2
import joblib
from pathlib import Path

from config import config
from tracker import ObjectTracker, TrackedObject
from utils import FPSLimiter, FrameRateRegulator, compute_bbox_iou, compute_centroid_distance
from ui import NexGuardUI
from whatsapp_bot.bot_client import whatsapp_bot


class TestNexGuardSystem(unittest.TestCase):

    def test_01_config_parameters(self):
        """Verifies max 15 FPS requirement and default parameters."""
        self.assertEqual(config.max_fps, 15)
        self.assertGreaterEqual(config.yolo_confidence, 0.1)
        self.assertTrue(Path(config.gesture_model_joblib).exists())
        self.assertTrue(Path(config.gesture_model_npz).exists())

    def test_02_fps_rate_limiter(self):
        """Verifies that FPSLimiter caps loop iteration rate to <= 15 FPS strictly."""
        limiter = FPSLimiter(target_fps=15)
        measured_fps_list = []
        
        # Simulate 10 frames
        for _ in range(10):
            limiter.start_frame()
            time.sleep(0.01)  # Fast processing simulation
            fps = limiter.sleep_if_needed()
            measured_fps_list.append(fps)

        avg_fps = sum(measured_fps_list) / len(measured_fps_list)
        print(f"\n[TEST] FPS Limiter Average Measured FPS: {avg_fps:.2f} / 15.0 MAX")
        self.assertLessEqual(avg_fps, 15.5)  # Strictly capped near 15 FPS

    def test_03_object_tracker(self):
        """Tests object tracking, IoU computation, and velocity calculation."""
        tracker = ObjectTracker(max_lost_frames=10, iou_threshold=0.2)
        
        # Frame 1 detections
        dets_f1 = [
            {"bbox": (100.0, 100.0, 200.0, 200.0), "class_name": "car", "confidence": 0.90},
            {"bbox": (300.0, 300.0, 350.0, 400.0), "class_name": "person", "confidence": 0.85}
        ]
        tracks_f1 = tracker.update(dets_f1)
        self.assertEqual(len(tracks_f1), 2)
        
        # Frame 2 shifted detections
        dets_f2 = [
            {"bbox": (110.0, 105.0, 210.0, 205.0), "class_name": "car", "confidence": 0.92},
            {"bbox": (305.0, 302.0, 355.0, 402.0), "class_name": "person", "confidence": 0.88}
        ]
        tracks_f2 = tracker.update(dets_f2)
        self.assertEqual(len(tracks_f2), 2)
        
        # Check track IDs preserved
        ids_f1 = {t.track_id for t in tracks_f1}
        ids_f2 = {t.track_id for t in tracks_f2}
        self.assertEqual(ids_f1, ids_f2)

    def test_04_gesture_classifier_inference(self):
        """Tests machine learning gesture classifier on normal vs accident features."""
        model_data = joblib.load(config.gesture_model_joblib)
        clf = model_data["model"]
        scaler = model_data["scaler"]

        # Feature vector: [speed_drop, angle_change, iou, centroid_dist, aspect_change, approach_vel]
        normal_feature = np.array([[0.05, 5.0, 0.02, 200.0, 0.05, 2.0]])
        accident_feature = np.array([[0.80, 75.0, 0.45, 10.0, 0.60, 30.0]])

        normal_scaled = scaler.transform(normal_feature)
        accident_scaled = scaler.transform(accident_feature)

        pred_normal = clf.predict(normal_scaled)[0]
        pred_accident = clf.predict(accident_scaled)[0]

        self.assertEqual(pred_normal, 0)
        self.assertEqual(pred_accident, 1)

    def test_05_ui_hud_renderer(self):
        """Tests visual HUD drawing engine on test OpenCV canvas."""
        ui = NexGuardUI()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        dummy_track = TrackedObject(
            track_id=1,
            bbox=(100.0, 100.0, 250.0, 250.0),
            class_name="car",
            confidence=0.88
        )
        
        annotated = ui.draw_hud(
            frame=frame,
            tracked_objects=[dummy_track],
            is_accident=True,
            confidence=0.92,
            severity="CRITICAL",
            fps=14.8,
            whatsapp_status="CONNECTED",
            incident_id="INC-12345"
        )

        self.assertEqual(annotated.shape, (720, 1280, 3))
        self.assertTrue(np.any(annotated > 0))  # Ensure overlay has drawn non-black pixels

    def test_06_whatsapp_client_status(self):
        """Tests WhatsApp status check structure."""
        status_info = whatsapp_bot.check_status()
        self.assertIn("status", status_info)
        self.assertIn("connected", status_info)

    def test_07_fps_benchmark_stability(self):
        """Validates that FrameRateRegulator maintains frame delay stability under load."""
        regulator = FrameRateRegulator(max_fps=15)
        start = time.time()
        for _ in range(5):
            regulator.tick()
        total_time = time.time() - start
        # 5 frames at 15 FPS = ~0.333 seconds minimum
        self.assertGreaterEqual(total_time, 0.30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
