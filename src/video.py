"""
NexGuard Video & Input Source Abstraction Module
Manages video streams from webcam, local video files, and static images.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from src.logger import get_logger
from src.utils import validate_file_path

logger = get_logger()


class InputSource:
    """Base class for video and image input sources."""

    def __init__(self, source_type: str, source_path: Any):
        self.source_type = source_type
        self.source_path = source_path
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.total_frames = 0
        self.is_opened = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release((self) -> None:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_path": str(self.source_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "is_opened": self.is_opened
        }


class WebcamInput(InputSource):
    """Webcam video capture input source."""

    def __init__(self, camera_index: int = 0):
        super().__init__("webcam", camera_index)
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_camera()

    def _init_camera(self) -> None:
        logger.info(f"Opening webcam camera index {self.camera_index}...")
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"[ERROR] Camera index {self.camera_index} could not be opened.")
                self.is_opened = False
                return

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.is_opened = True
            logger.info(f"[✓] Webcam initialized ({self.width}x{self.height} @ {self.fps:.1f} FPS)")
        except Exception as e:
            logger.error(f"[ERROR] Camera initialization failure: {e}")
            self.is_opened = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("Webcam input released.")


class VideoFileInput(InputSource):
    """Video file input source."""

    def __init__(self, video_path: str):
        super().__init__("video", video_path)
        self.video_path = str(video_path).strip()
        self.cap: Optional[cv2.VideoCapture] = None
        self._init_video()

    def _init_video(self) -> None:
        if not validate_file_path(self.video_path):
            logger.error(f"[ERROR] Video file does not exist or is invalid: '{self.video_path}'")
            self.is_opened = False
            return

        logger.info(f"Opening video file '{self.video_path}'...")
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logger.error(f"[ERROR] Failed to open video file: '{self.video_path}'")
                self.is_opened = False
                return

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.is_opened = True
            logger.info(f"[✓] Video loaded ({self.width}x{self.height}, {self.total_frames} frames @ {self.fps:.1f} FPS)")
        except Exception as e:
            logger.error(f"[ERROR] Video file load error: {e}")
            self.is_opened = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("Video file input released.")


class ImageFileInput(InputSource):
    """Static image file input source."""

    def __init__(self, image_path: str):
        super().__init__("image", image_path)
        self.image_path = str(image_path).strip()
        self.image_frame: Optional[np.ndarray] = None
        self._read_image = False
        self._init_image()

    def _init_image(self) -> None:
        if not validate_file_path(self.image_path):
            logger.error(f"[ERROR] Image file does not exist or is invalid: '{self.image_path}'")
            self.is_opened = False
            return

        logger.info(f"Loading image file '{self.image_path}'...")
        try:
            self.image_frame = cv2.imread(self.image_path)
            if self.image_frame is None:
                logger.error(f"[ERROR] Could not decode image file: '{self.image_path}'")
                self.is_opened = False
                return

            self.height, self.width = self.image_frame.shape[:2]
            self.total_frames = 1
            self.is_opened = True
            logger.info(f"[✓] Image loaded ({self.width}x{self.height})")
        except Exception as e:
            logger.error(f"[ERROR] Image loading error: {e}")
            self.is_opened = False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.image_frame is None:
            return False, None
        # Return frame continuously for interactive display loop
        return True, self.image_frame.copy()

    def release(self) -> None:
        self.image_frame = None
        self.is_opened = False
        logger.info("Image file input released.")
