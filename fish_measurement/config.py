import cv2
import numpy as np
from pupil_apriltags import Detector

TAG_SIZE_MM = 100.0
MAX_WIDTH = 1000
MAX_HEIGHT = 700
MIN_OBJECT_AREA_PX = 1800.0
MAX_OBJECT_AREA_RATIO = 0.55
TAG_IGNORE_PAD_FACTOR = 0.35
BORDER_REJECT_PX = 6
MIN_ASPECT_RATIO = 2.2
MIN_PERIMETER_PX = 180.0
MAX_BACKGROUND_AREA_RATIO = 0.80

WATER_PROFILES = {
    "clear": {
        "clahe_clip": 1.8,
        "clahe_grid": 8,
        "blur_ksize": 5,
        "canny_low": 60,
        "canny_high": 170,
        "adaptive_block_size": 31,
        "adaptive_c": 5,
        "color_s_min": 45,
        "color_v_min": 35,
        "color_hue_low_max": 45,
        "color_hue_high_min": 150,
        "color_open_kernel": 5,
        "color_close_kernel": 13,
        "min_color_overlap_ratio": 0.12,
        "min_object_area_px": MIN_OBJECT_AREA_PX,
        "min_perimeter_px": MIN_PERIMETER_PX,
        "min_aspect_ratio": MIN_ASPECT_RATIO,
        "max_aspect_ratio": 9.0,
        "min_length_mm": 40.0,
        "max_length_mm": 1200.0,
        "min_solidity": 0.35,
        "min_length_px_factor": 0.45,
        "min_thickness_px_factor": 0.10,
        "min_area_tag_ratio": 0.10,
        "bottom_ignore_ratio": 0.10,
        "tag_min_side_px": 24.0,
        "tag_expected_x_max_ratio": 0.45,
        "tag_expected_y_min_ratio": 0.55,
        "tag_location_weight": 14.0,
        "object_min_saturation": 18.0,
        "object_min_value": 28.0,
        "min_extent": 0.12,
    },
    "murky": {
        "clahe_clip": 3.0,
        "clahe_grid": 8,
        "blur_ksize": 7,
        "canny_low": 35,
        "canny_high": 120,
        "adaptive_block_size": 41,
        "adaptive_c": 3,
        "color_s_min": 20,
        "color_v_min": 22,
        "color_hue_low_max": 50,
        "color_hue_high_min": 145,
        "color_open_kernel": 3,
        "color_close_kernel": 17,
        "min_color_overlap_ratio": 0.08,
        "min_object_area_px": 1400.0,
        "min_perimeter_px": 140.0,
        "min_aspect_ratio": 1.9,
        "max_aspect_ratio": 10.5,
        "min_length_mm": 35.0,
        "max_length_mm": 1200.0,
        "min_solidity": 0.30,
        "min_length_px_factor": 0.40,
        "min_thickness_px_factor": 0.08,
        "min_area_tag_ratio": 0.08,
        "bottom_ignore_ratio": 0.14,
        "tag_min_side_px": 20.0,
        "tag_expected_x_max_ratio": 0.50,
        "tag_expected_y_min_ratio": 0.50,
        "tag_location_weight": 10.0,
        "object_min_saturation": 14.0,
        "object_min_value": 22.0,
        "min_extent": 0.10,
    },
}

APRILTAG_DETECTORS = [
    Detector(
        families="tag36h11 tag25h9 tag16h5",
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.8,
        refine_edges=1,
        decode_sharpening=0.5,
    ),
    Detector(
        families="tag36h11 tag25h9 tag16h5",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.8,
    ),
]


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def get_profile(profile_name: str) -> dict:
    if profile_name not in WATER_PROFILES:
        raise ValueError(f"Unsupported water profile: {profile_name}")
    return WATER_PROFILES[profile_name]


def select_water_profile_auto(frame: np.ndarray) -> str:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat_median = float(np.median(hsv[:, :, 1]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast_std = float(np.std(gray))
    if sat_median < 38.0 or contrast_std < 45.0:
        return "murky"
    return "clear"

