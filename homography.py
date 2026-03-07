import cv2
import numpy as np
import os
import time
import argparse
from pathlib import Path
from pupil_apriltags import Detector

# ================= CONFIG =================
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
# =========================================

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


def preprocess_gray(gray: np.ndarray, profile: dict) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(profile["clahe_clip"]),
        tileGridSize=(int(profile["clahe_grid"]), int(profile["clahe_grid"])),
    )
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 40, 40)
    blur_k = ensure_odd(int(profile["blur_ksize"]))
    return cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)


def detect_apriltag(gray_candidates: list, image_shape: tuple, profile: dict) -> tuple:
    best_detection = None
    best_scale = 1.0
    best_score = -1e9

    height, width = image_shape[:2]
    x_limit = float(profile["tag_expected_x_max_ratio"]) * width
    y_limit = float(profile["tag_expected_y_min_ratio"]) * height
    location_weight = float(profile["tag_location_weight"])
    min_side_px = float(profile["tag_min_side_px"])

    for gray, scale in gray_candidates:
        thresholded = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(thresholded)
        detection_inputs = [gray, thresholded, inverted, otsu]

        for image in detection_inputs:
            for detector in APRILTAG_DETECTORS:
                detections = detector.detect(image)
                for det in detections:
                    corners = det.corners.astype(np.float32) / float(scale)
                    side_px = float(
                        np.mean(
                            [
                                np.linalg.norm(corners[i] - corners[(i + 1) % 4])
                                for i in range(4)
                            ]
                        )
                    )
                    if side_px < min_side_px:
                        continue
                    center = np.mean(corners, axis=0)
                    in_expected_region = center[0] <= x_limit and center[1] >= y_limit

                    hamming = int(getattr(det, "hamming", 99))
                    decision_margin = float(getattr(det, "decision_margin", 0.0))
                    score = decision_margin - (20.0 * hamming) + (0.04 * side_px)
                    score += location_weight if in_expected_region else (-0.6 * location_weight)

                    if score > best_score:
                        best_score = score
                        best_detection = det
                        best_scale = float(scale)

    return best_detection, best_scale


def max_diameter_from_points(points_xy: np.ndarray):
    if points_xy.shape[0] < 2:
        return None

    hull = cv2.convexHull(points_xy.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
    if hull.shape[0] < 2:
        return None

    if hull.shape[0] > 500:
        step = int(np.ceil(hull.shape[0] / 500.0))
        hull = hull[::step]

    diffs = hull[:, None, :] - hull[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    flat_idx = int(np.argmax(dist2))
    i, j = np.unravel_index(flat_idx, dist2.shape)
    p1 = hull[i]
    p2 = hull[j]
    return float(np.sqrt(dist2[i, j])), p1, p2


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
    # Emphasize warm fish hues (orange/red/yellow) and suppress cyan water/background.
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


def analyze_image(
    image_path: str,
    visualize: bool = False,
    wait_for_key: bool = False,
    water_profile: str = "auto",
    save_overlay_dir: str | None = None,
) -> dict:
    result = {
        "image_path": image_path,
        "water_profile": water_profile,
        "tag_detected": False,
        "segmentation_mode": "none",
        "detected_contours": 0,
        "fish_count": 0,
        "lengths_mm": [],
        "continue_processing": True,
        "overlay_path": None,
    }

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if water_profile == "auto":
        selected_profile_name = select_water_profile_auto(original_image)
    else:
        selected_profile_name = water_profile
    profile = get_profile(selected_profile_name)
    result["water_profile"] = selected_profile_name

    h, w = original_image.shape[:2]
    disp_scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
    disp_w, disp_h = int(w * disp_scale), int(h * disp_scale)

    gray_raw = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray_raw)
    gray_preprocessed = preprocess_gray(gray_raw, profile)
    gray_candidates = [
        (gray_raw, 1.0),
        (gray_eq, 1.0),
        (gray_preprocessed, 1.0),
        (cv2.resize(gray_raw, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), 1.5),
        (cv2.resize(gray_eq, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), 1.5),
        (cv2.resize(gray_preprocessed, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), 1.5),
    ]

    det, det_scale = detect_apriltag(gray_candidates, original_image.shape, profile)
    if det is None:
        if save_overlay_dir:
            out_dir = Path(save_overlay_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(image_path).stem}_measurement.jpg"
            no_tag_overlay = original_image.copy()
            cv2.putText(
                no_tag_overlay,
                "No AprilTag detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(str(out_path), no_tag_overlay)
            result["overlay_path"] = str(out_path)
        if visualize:
            cv2.imshow("Measurement", cv2.resize(original_image, (disp_w, disp_h)))
            if wait_for_key:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    result["continue_processing"] = False
            else:
                cv2.waitKey(1)
            cv2.destroyAllWindows()
        return result

    corners = det.corners.astype(np.float32) / float(det_scale)
    result["tag_detected"] = True

    dst_mm = np.array(
        [[0, 0], [TAG_SIZE_MM, 0], [TAG_SIZE_MM, TAG_SIZE_MM], [0, TAG_SIZE_MM]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst_mm)
    H_inv = np.linalg.inv(H)

    # Build ignore mask around the tag in original image.
    tag_side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    tag_area_px = float(cv2.contourArea(corners.astype(np.float32)))
    pad_px = max(6, int(tag_side_px * TAG_IGNORE_PAD_FACTOR))
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ignore_mask, [corners.astype(np.int32)], 255)
    ignore_mask = cv2.dilate(ignore_mask, np.ones((pad_px, pad_px), np.uint8), iterations=1)
    bottom_ignore_ratio = float(profile["bottom_ignore_ratio"])
    if bottom_ignore_ratio > 0.0:
        y_start = int(h * (1.0 - bottom_ignore_ratio))
        ignore_mask[y_start:, :] = 255

    max_area_px = MAX_OBJECT_AREA_RATIO * (w * h)
    bg_area_px = MAX_BACKGROUND_AREA_RATIO * (w * h)
    min_color_overlap_ratio = float(profile["min_color_overlap_ratio"])
    contour_sets = [
        ("edge", detect_object_contours(original_image, ignore_mask, profile), 0.0),
        ("color", detect_color_object_contours(original_image, ignore_mask, profile), min_color_overlap_ratio),
        ("body", detect_adaptive_body_contours(original_image, ignore_mask, profile), 0.0),
    ]
    fish_color_mask = build_fish_color_mask(original_image, ignore_mask, profile)

    min_object_area_px = float(profile["min_object_area_px"])
    min_aspect_ratio = float(profile["min_aspect_ratio"])
    max_aspect_ratio = float(profile["max_aspect_ratio"])
    min_perimeter_px = float(profile["min_perimeter_px"])
    min_length_mm = float(profile["min_length_mm"])
    max_length_mm = float(profile["max_length_mm"])
    min_solidity = float(profile["min_solidity"])
    min_length_px_factor = float(profile["min_length_px_factor"])
    min_thickness_px_factor = float(profile["min_thickness_px_factor"])
    min_area_tag_ratio = float(profile["min_area_tag_ratio"])
    object_min_saturation = float(profile["object_min_saturation"])
    object_min_value = float(profile["object_min_value"])
    min_extent = float(profile["min_extent"])
    min_area_px_dynamic = max(min_object_area_px, min_area_tag_ratio * tag_area_px)
    def extract_measurements(
        contours: list,
        area_min_px: float,
        aspect_min: float,
        aspect_max: float,
        perimeter_min_px: float,
        solidity_min: float,
        length_min_mm: float,
        length_max_mm: float,
        length_min_px_factor_local: float,
        thickness_min_px_factor_local: float,
        color_overlap_min_local: float,
    ) -> list:
        prefiltered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < area_min_px or area > max_area_px:
                continue
            if area > bg_area_px:
                continue
            if contour_touches_border(cnt, w, h, BORDER_REJECT_PX):
                continue
            if contour_overlap_ratio_with_mask(cnt, ignore_mask) > 0.02:
                continue
            if contour_overlap_ratio_with_mask(cnt, fish_color_mask) < color_overlap_min_local:
                continue
            _, mean_s, mean_v = contour_mean_hsv(original_image, cnt)
            if (mean_s < object_min_saturation) and (mean_v < object_min_value):
                continue
            prefiltered.append(cnt)

        if not prefiltered:
            return []

        # Merge nearby fragments to recover full fish body/tail before measurement.
        merge_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(merge_mask, prefiltered, -1, 255, -1)
        merge_k = ensure_odd(max(5, int(0.16 * tag_side_px)))
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (merge_k, merge_k))
        merge_mask = cv2.morphologyEx(merge_mask, cv2.MORPH_CLOSE, merge_kernel, iterations=1)
        merged_contours, _ = cv2.findContours(merge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_measurements = []
        for cnt in merged_contours:
            area = cv2.contourArea(cnt)
            if area < area_min_px or area > max_area_px:
                continue
            aspect = contour_aspect_ratio(cnt)
            if aspect < aspect_min or aspect > aspect_max:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < perimeter_min_px:
                continue
            solidity = contour_solidity(cnt)
            if solidity < solidity_min:
                continue
            extent = contour_extent(cnt)
            if extent < min_extent:
                continue

            cnt_mm = cv2.perspectiveTransform(cnt.astype(np.float32), H).reshape(-1, 2)
            axis_mm = max_diameter_from_points(cnt_mm)
            if axis_mm is None:
                continue

            length_mm, p1_mm, p2_mm = axis_mm
            if length_mm < length_min_mm or length_mm > length_max_mm:
                continue

            pair_mm = np.array([[p1_mm, p2_mm]], dtype=np.float32)
            pair_px = cv2.perspectiveTransform(pair_mm, H_inv)[0]
            length_px = float(np.linalg.norm(pair_px[0] - pair_px[1]))
            if length_px < (length_min_px_factor_local * tag_side_px):
                continue

            rw, rh = cv2.minAreaRect(cnt)[1]
            thickness_px = float(min(rw, rh))
            if thickness_px < (thickness_min_px_factor_local * tag_side_px):
                continue

            p1 = tuple(pair_px[0].astype(int))
            p2 = tuple(pair_px[1].astype(int))
            fishness = (
                (0.35 * min(1.0, length_px / max(1.0, 1.2 * tag_side_px)))
                + (0.25 * min(1.0, area / max(1.0, 0.45 * tag_area_px)))
                + (0.20 * min(1.0, solidity))
                + (0.20 * min(1.0, extent))
            )
            local_measurements.append((length_mm, cnt, p1, p2, fishness))

        local_measurements.sort(key=lambda x: (x[4], x[0]), reverse=True)
        local_measurements = [m for m in local_measurements if m[4] >= 0.42]
        local_measurements = [(m[0], m[1], m[2], m[3]) for m in local_measurements]
        return local_measurements

    pass_params = [
        (
            min_area_px_dynamic,
            min_aspect_ratio,
            max_aspect_ratio,
            min_perimeter_px,
            min_solidity,
            min_length_mm,
            max_length_mm,
            min_length_px_factor,
            min_thickness_px_factor,
            0.0,
        ),
        (
            0.45 * min_area_px_dynamic,
            0.75 * min_aspect_ratio,
            1.35 * max_aspect_ratio,
            0.60 * min_perimeter_px,
            0.70 * min_solidity,
            0.80 * min_length_mm,
            max_length_mm,
            0.75 * min_length_px_factor,
            0.70 * min_thickness_px_factor,
            0.0,
        ),
    ]

    chosen_mode = "none"
    chosen_contour_count = 0
    measurements = []
    best_score = (-1.0, -1, -1.0)
    for pass_index, params in enumerate(pass_params):
        for mode_name, contours, mode_overlap_min in contour_sets:
            params_with_mode = params[:-1] + (max(params[-1], mode_overlap_min),)
            local_measurements = extract_measurements(contours, *params_with_mode)
            if local_measurements:
                max_len = float(local_measurements[0][0])
                total_len = float(sum(item[0] for item in local_measurements))
                # Prefer strict pass when quality is comparable.
                strict_bonus = 1000.0 if pass_index == 0 else 0.0
                area_sum = float(sum(cv2.contourArea(item[1]) for item in local_measurements))
                score = (strict_bonus + max_len, len(local_measurements), total_len + 0.0005 * area_sum)
            else:
                score = (-1.0, -1, -1.0)

            if score > best_score:
                best_score = score
                measurements = local_measurements
                chosen_mode = mode_name if pass_index == 0 else f"{mode_name}_relaxed"
                chosen_contour_count = len(contours)
        if measurements:
            break

    # Generalized cleanup: remove tiny residual blobs compared with dominant fish contours.
    if measurements:
        contour_areas = [cv2.contourArea(item[1]) for item in measurements]
        max_area = max(contour_areas) if contour_areas else 0.0
        if max_area > 0.0:
            area_keep_ratio = 0.35
            measurements = [
                item for item in measurements if cv2.contourArea(item[1]) >= (area_keep_ratio * max_area)
            ]
            measurements.sort(key=lambda x: x[0], reverse=True)

    result["segmentation_mode"] = chosen_mode
    result["detected_contours"] = chosen_contour_count
    result["fish_count"] = len(measurements)
    result["lengths_mm"] = [float(item[0]) for item in measurements]

    overlay = original_image.copy()
    # Draw tag.
    for i in range(4):
        cv2.line(
            overlay,
            tuple(corners[i].astype(int)),
            tuple(corners[(i + 1) % 4].astype(int)),
            (0, 255, 0),
            2,
        )
    center = np.mean(corners, axis=0)
    cv2.putText(
        overlay,
        f"Tag {det.tag_id}",
        (int(center[0]), int(center[1]) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    for idx, (length_mm, cnt, p1, p2) in enumerate(measurements, start=1):
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 255), 2)
        cv2.line(overlay, p1, p2, (255, 0, 0), 2)
        cv2.circle(overlay, p1, 4, (0, 0, 255), -1)
        cv2.circle(overlay, p2, 4, (0, 0, 255), -1)
        lx = min(p1[0], p2[0])
        ly = min(p1[1], p2[1]) - 8
        cv2.putText(
            overlay,
            f"Obj{idx}: {length_mm:.1f} mm",
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    if save_overlay_dir:
        out_dir = Path(save_overlay_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(image_path).stem}_measurement.jpg"
        cv2.imwrite(str(out_path), overlay)
        result["overlay_path"] = str(out_path)

    if not visualize:
        return result

    overlay_disp = cv2.resize(overlay, (disp_w, disp_h))
    cv2.imshow("Measurement", overlay_disp)
    if wait_for_key:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            result["continue_processing"] = False
    else:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    return result


def process_image(
    image_path: str,
    visualize: bool = True,
    wait_for_key: bool = True,
    water_profile: str = "auto",
) -> bool:
    print(f"\nProcessing: {image_path}")
    analysis = analyze_image(
        image_path,
        visualize=visualize,
        wait_for_key=wait_for_key,
        water_profile=water_profile,
    )

    if not analysis["tag_detected"]:
        print("No AprilTag detected")
        if visualize and wait_for_key:
            print("Press any key to continue (q to quit).")
        return analysis["continue_processing"]

    print(f"Segmentation mode: {analysis['segmentation_mode']}")
    print(
        f"Detected contours: {analysis['detected_contours']} | "
        f"Measured objects: {analysis['fish_count']}"
    )
    if analysis["lengths_mm"]:
        print("\nAuto-detected object lengths:")
        for i, length_mm in enumerate(analysis["lengths_mm"], start=1):
            print(f"  Obj{i}: {length_mm:.2f} mm")
    else:
        print("No objects detected. Try lowering MIN_OBJECT_AREA_PX or TAG_IGNORE_PAD_FACTOR.")

    if visualize and wait_for_key:
        print("Press any key to continue (q to quit).")
    return analysis["continue_processing"]


def process_directory_loop(
    input_dir: str = "images",
    interval_seconds: float = 2.0,
    extensions=(".jpg", ".jpeg", ".png"),
    visualize: bool = False,
    water_profile: str = "auto",
) -> None:
    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)
    processed_files = set()

    print(f"Watching directory: {input_path.resolve()}")
    print(f"Allowed extensions: {', '.join(extensions)}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            files = sorted(
                [
                    p
                    for p in input_path.iterdir()
                    if p.is_file()
                    and not p.name.startswith(".")
                    and not p.name.startswith("._")
                    and p.suffix.lower() in extensions
                ],
                key=lambda p: p.stat().st_mtime,
            )

            found_new = False
            for file_path in files:
                if file_path in processed_files:
                    continue
                found_new = True
                print(f"\nAuto-processing: {file_path}")
                try:
                    process_image(
                        str(file_path),
                        visualize=visualize,
                        wait_for_key=False,
                        water_profile=water_profile,
                    )
                except Exception as exc:
                    print(f"Failed to process {file_path}: {exc}")
                finally:
                    processed_files.add(file_path)

            if not found_new:
                print("No new images found. Waiting...")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped directory watcher.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homography-based fish measurement")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically process new images from a directory loop",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="images",
        help="Input directory used in --auto mode",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=2.0,
        help="Polling interval used in --auto mode",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png",
        help="Comma-separated file extensions for --auto mode",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show OpenCV visualization window (optional for Pi with desktop)",
    )
    parser.add_argument(
        "--water-profile",
        choices=["auto"] + sorted(WATER_PROFILES.keys()),
        default="auto",
        help="Segmentation profile tuned for water conditions.",
    )
    args = parser.parse_args()

    if args.auto:
        exts = tuple(
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in args.extensions.split(",")
            if ext.strip()
        )
        if not exts:
            raise ValueError("No valid extensions provided.")
        process_directory_loop(
            input_dir=args.input_dir,
            interval_seconds=args.interval_seconds,
            extensions=exts,
            visualize=args.visualize,
            water_profile=args.water_profile,
        )
    else:
        if not args.image:
            raise ValueError("Provide --image for single-image mode or use --auto.")
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            print(f"Current directory: {os.getcwd()}")
        else:
            process_image(
                args.image,
                visualize=True,
                wait_for_key=True,
                water_profile=args.water_profile,
            )
