import cv2
import numpy as np

from .config import ensure_odd


def preprocess_gray(gray: np.ndarray, profile: dict) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(profile["clahe_clip"]),
        tileGridSize=(int(profile["clahe_grid"]), int(profile["clahe_grid"])),
    )
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 40, 40)
    blur_k = ensure_odd(int(profile["blur_ksize"]))
    return cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)


def contour_overlap_ratio_with_mask(contour: np.ndarray, mask: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return 0.0
    roi = mask[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    cnt_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0, 0] -= x
    shifted[:, 0, 1] -= y
    cv2.drawContours(cnt_mask, [shifted], -1, 255, -1)

    denom = float(np.count_nonzero(cnt_mask))
    if denom <= 0:
        return 0.0
    overlap = cv2.bitwise_and(cnt_mask, roi)
    return float(np.count_nonzero(overlap)) / denom


def contour_touches_border(contour: np.ndarray, width: int, height: int, margin: int) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    if x <= margin or y <= margin:
        return True
    if (x + w) >= (width - margin) or (y + h) >= (height - margin):
        return True
    return False


def contour_aspect_ratio(contour: np.ndarray) -> float:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    lo = min(w, h)
    hi = max(w, h)
    if lo <= 1e-6:
        return 0.0
    return float(hi / lo)


def contour_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area <= 1e-6:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1e-6:
        return 0.0
    return float(area / hull_area)


def contour_extent(contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    box_area = float(w * h)
    if box_area <= 1e-6:
        return 0.0
    area = cv2.contourArea(contour)
    return float(area / box_area)


def contour_mean_hsv(frame: np.ndarray, contour: np.ndarray) -> tuple:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)
    return float(mean_h), float(mean_s), float(mean_v)


def detect_adaptive_body_contours(frame: np.ndarray, ignore_mask: np.ndarray, profile: dict) -> list:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = preprocess_gray(gray, profile)
    edges = cv2.Canny(blur, int(profile["canny_low"]), int(profile["canny_high"]))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    s_thr = max(int(np.percentile(s, 62)), int(profile["object_min_saturation"]))
    v_blur = cv2.GaussianBlur(v, (21, 21), 0)
    local_contrast = cv2.absdiff(v, v_blur)
    contrast_mask = (local_contrast > 14).astype(np.uint8) * 255
    sat_mask = (s > s_thr).astype(np.uint8) * 255

    combined = cv2.bitwise_or(edges, sat_mask)
    combined = cv2.bitwise_or(combined, contrast_mask)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=2
    )
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    combined[ignore_mask > 0] = 0
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_object_contours(frame: np.ndarray, ignore_mask: np.ndarray, profile: dict) -> list:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = preprocess_gray(gray, profile)

    edges = cv2.Canny(blur, int(profile["canny_low"]), int(profile["canny_high"]))
    block_size = ensure_odd(int(profile["adaptive_block_size"]))
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        int(profile["adaptive_c"]),
    )
    combined = cv2.bitwise_or(edges, thresh)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
    )
    combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    combined[ignore_mask > 0] = 0

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def build_fish_color_mask(frame: np.ndarray, ignore_mask: np.ndarray, profile: dict) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hue_low_max = int(profile["color_hue_low_max"])
    hue_high_min = int(profile["color_hue_high_min"])
    color_mask = (
        (s > int(profile["color_s_min"]))
        & ((h <= hue_low_max) | (h >= hue_high_min))
        & (v > int(profile["color_v_min"]))
    ).astype(np.uint8) * 255
    open_k = int(profile["color_open_kernel"])
    close_k = int(profile["color_close_kernel"])
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8), iterations=1
    )
    color_mask = cv2.morphologyEx(
        color_mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8), iterations=2
    )
    color_mask[ignore_mask > 0] = 0
    return color_mask


def detect_color_object_contours(frame: np.ndarray, ignore_mask: np.ndarray, profile: dict) -> list:
    color_mask = build_fish_color_mask(frame, ignore_mask, profile)
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

