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
        "color_open_kernel": 5,
        "color_close_kernel": 13,
        "min_object_area_px": MIN_OBJECT_AREA_PX,
        "min_perimeter_px": MIN_PERIMETER_PX,
        "min_aspect_ratio": MIN_ASPECT_RATIO,
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
        "color_open_kernel": 3,
        "color_close_kernel": 17,
        "min_object_area_px": 1400.0,
        "min_perimeter_px": 140.0,
        "min_aspect_ratio": 1.9,
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


def preprocess_gray(gray: np.ndarray, profile: dict) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=float(profile["clahe_clip"]),
        tileGridSize=(int(profile["clahe_grid"]), int(profile["clahe_grid"])),
    )
    enhanced = clahe.apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 7, 40, 40)
    blur_k = ensure_odd(int(profile["blur_ksize"]))
    return cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)


def detect_apriltag(gray: np.ndarray) -> list:
    thresholded = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    inverted = cv2.bitwise_not(thresholded)
    detection_inputs = [gray, thresholded, inverted]

    for image in detection_inputs:
        for detector in APRILTAG_DETECTORS:
            detections = detector.detect(image)
            if detections:
                return detections
    return []


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


def detect_color_object_contours(frame: np.ndarray, ignore_mask: np.ndarray, profile: dict) -> list:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Keep saturated non-brown colors (helps isolate fish/tools from wood-like backgrounds).
    color_mask = (
        (s > int(profile["color_s_min"]))
        & ((h < 5) | (h > 30))
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
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def analyze_image(
    image_path: str,
    visualize: bool = False,
    wait_for_key: bool = False,
    water_profile: str = "murky",
) -> dict:
    profile = get_profile(water_profile)
    result = {
        "image_path": image_path,
        "water_profile": water_profile,
        "tag_detected": False,
        "segmentation_mode": "none",
        "detected_contours": 0,
        "fish_count": 0,
        "lengths_mm": [],
        "continue_processing": True,
    }

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = original_image.shape[:2]
    disp_scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
    disp_w, disp_h = int(w * disp_scale), int(h * disp_scale)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = preprocess_gray(gray, profile)

    detections = detect_apriltag(gray)
    if not detections:
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

    det = detections[0]
    corners = det.corners.astype(np.float32)
    result["tag_detected"] = True

    dst_mm = np.array(
        [[0, 0], [TAG_SIZE_MM, 0], [TAG_SIZE_MM, TAG_SIZE_MM], [0, TAG_SIZE_MM]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst_mm)
    H_inv = np.linalg.inv(H)

    # Build ignore mask around the tag in original image.
    tag_side_px = float(np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]))
    pad_px = max(6, int(tag_side_px * TAG_IGNORE_PAD_FACTOR))
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(ignore_mask, [corners.astype(np.int32)], 255)
    ignore_mask = cv2.dilate(ignore_mask, np.ones((pad_px, pad_px), np.uint8), iterations=1)

    max_area_px = MAX_OBJECT_AREA_RATIO * (w * h)
    bg_area_px = MAX_BACKGROUND_AREA_RATIO * (w * h)
    contour_sets = [
        ("edge", detect_object_contours(original_image, ignore_mask, profile)),
        ("color", detect_color_object_contours(original_image, ignore_mask, profile)),
    ]

    min_object_area_px = float(profile["min_object_area_px"])
    min_aspect_ratio = float(profile["min_aspect_ratio"])
    min_perimeter_px = float(profile["min_perimeter_px"])
    chosen_mode = "none"
    chosen_contours = []
    measurements = []
    for mode_name, contours in contour_sets:
        local_measurements = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_object_area_px or area > max_area_px:
                continue
            if area > bg_area_px:
                continue
            touches_border = contour_touches_border(cnt, w, h, BORDER_REJECT_PX)
            aspect = contour_aspect_ratio(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if touches_border and (aspect < (min_aspect_ratio + 0.8) or perimeter < (min_perimeter_px + 80.0)):
                continue
            if contour_overlap_ratio_with_mask(cnt, ignore_mask) > 0.02:
                continue
            if aspect < min_aspect_ratio:
                continue
            if perimeter < min_perimeter_px:
                continue

            cnt_mm = cv2.perspectiveTransform(cnt.astype(np.float32), H).reshape(-1, 2)
            axis_mm = max_diameter_from_points(cnt_mm)
            if axis_mm is None:
                continue

            length_mm, p1_mm, p2_mm = axis_mm
            pair_mm = np.array([[p1_mm, p2_mm]], dtype=np.float32)
            pair_px = cv2.perspectiveTransform(pair_mm, H_inv)[0]
            p1 = tuple(pair_px[0].astype(int))
            p2 = tuple(pair_px[1].astype(int))
            local_measurements.append((length_mm, cnt, p1, p2))

        if len(local_measurements) > len(measurements):
            measurements = local_measurements
            chosen_mode = mode_name
            chosen_contours = contours

    measurements.sort(key=lambda x: x[0], reverse=True)
    result["segmentation_mode"] = chosen_mode
    result["detected_contours"] = len(chosen_contours)
    result["fish_count"] = len(measurements)
    result["lengths_mm"] = [float(item[0]) for item in measurements]

    if not visualize:
        return result

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
    water_profile: str = "murky",
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
    water_profile: str = "murky",
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
        choices=sorted(WATER_PROFILES.keys()),
        default="murky",
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
