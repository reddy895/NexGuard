"""
NexGuard — Training Augmentations Pipeline
=========================================
Applies semantically sound data augmentations for CCTV road accident detection:
- Preserves collision evidence and ground-truth meaning
- Prohibits vertical flips (traffic is directional)
- Simulates CCTV motion blur and sensor noise
- Enforces strict zero-augmentation on validation and test sets
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

@dataclass
class AugmentationConfig:
    fliplr: float = 0.50
    degrees: float = 5.0
    translate: float = 0.08
    scale: float = 0.10
    hsv_h: float = 0.015
    hsv_s: float = 0.40
    hsv_v: float = 0.30
    motion_blur_prob: float = 0.15
    sensor_noise_prob: float = 0.10
    augment_eval_data: bool = False

class CCTVImageAugmentor:
    """
    Applies image transforms tailored for surveillance traffic streams.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None) -> None:
        self.cfg = config or AugmentationConfig()

    def apply_motion_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def apply_sensor_noise(self, image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
        noise = np.random.normal(0, sigma, image.shape)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def augment(self, image: np.ndarray, is_training: bool = True) -> np.ndarray:
        if not is_training or self.cfg.augment_eval_data:
            return image

        aug = image.copy()
        # 1. Horizontal flip
        if random.random() < self.cfg.fliplr:
            aug = cv2.flip(aug, 1)

        # 2. Motion blur
        if random.random() < self.cfg.motion_blur_prob:
            aug = self.apply_motion_blur(aug)

        # 3. Sensor noise
        if random.random() < self.cfg.sensor_noise_prob:
            aug = self.apply_sensor_noise(aug)

        return aug
