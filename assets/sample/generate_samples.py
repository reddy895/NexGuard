"""
NexGuard Sample Asset Generator
Generates sample synthetic video and image files for testing and offline execution.
"""

import cv2
import numpy as np
from pathlib import Path

SAMPLE_DIR = Path("assets/sample")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_IMAGE_PATH = SAMPLE_DIR / "traffic_sample.jpg"
SAMPLE_VIDEO_PATH = SAMPLE_DIR / "surveillance_sample.mp4"


def create_sample_image(path: Path = SAMPLE_IMAGE_PATH) -> Path:
    """Generates a synthetic traffic scene image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw road background
    cv2.rectangle(img, (0, 240), (640, 480), (60, 60, 60), -1)
    # Draw road lane markers
    for x in range(20, 640, 80):
        cv2.rectangle(img, (x, 350), (x + 40, 355), (255, 255, 255), -1)

    # Draw synthetic vehicles and pedestrian shapes
    # Vehicle 1 (Car)
    cv2.rectangle(img, (100, 300), (240, 380), (0, 0, 200), -1)
    cv2.putText(img, "Car 1", (110, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Vehicle 2 (Truck)
    cv2.rectangle(img, (320, 260), (520, 370), (200, 100, 0), -1)
    cv2.putText(img, "Truck 1", (340, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Pedestrian shape
    cv2.circle(img, (70, 280), 12, (180, 200, 255), -1)
    cv2.rectangle(img, (60, 292), (80, 340), (180, 200, 255), -1)

    cv2.imwrite(str(path), img)
    return path


def create_sample_video(path: Path = SAMPLE_VIDEO_PATH, num_frames: int = 150) -> Path:
    """Generates a synthetic surveillance video stream."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 30.0, (640, 480))

    for frame_idx in range(num_frames):
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Background road
        cv2.rectangle(img, (0, 200), (640, 480), (50, 50, 50), -1)
        # Moving road markers
        offset = (frame_idx * 5) % 80
        for x in range(-80 + offset, 640, 80):
            cv2.rectangle(img, (max(0, x), 340), (min(640, x + 40), 345), (255, 255, 255), -1)

        # Moving car
        car_x = (frame_idx * 4) % 640
        cv2.rectangle(img, (car_x, 280), (car_x + 120, 350), (0, 165, 255), -1)
        cv2.putText(img, "Vehicle", (car_x + 10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(img)

    out.release()
    return path


if __name__ == "__main__":
    print("Generating sample assets...")
    img_p = create_sample_image()
    print(f"Sample image created: {img_p}")
    vid_p = create_sample_video()
    print(f"Sample video created: {vid_p}")
