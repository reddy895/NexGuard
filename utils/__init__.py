"""
NexGuard Utilities Package
"""

from utils.logger import setup_logger, logger
from utils.geometry import (
    get_bbox_center,
    get_bbox_dimensions,
    get_bbox_area,
    compute_euclidean_distance,
    compute_iou,
    compute_vector_magnitude,
    compute_angle_degrees,
    compute_angle_change
)
from utils.validation import (
    validate_image_path,
    validate_video_path,
    validate_webcam_index,
    validate_rtsp_url,
    validate_accident_dataset
)
from utils.video import VideoStreamHandler, resize_frame
