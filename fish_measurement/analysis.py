import cv2
import numpy as np
import time
from pathlib import Path

from .apriltag import detect_apriltag
from .config import (
    BORDER_REJECT_PX,
    MAX_BACKGROUND_AREA_RATIO,
    MAX_HEIGHT,
    MAX_OBJECT_AREA_RATIO,
    MAX_WIDTH,
    TAG_IGNORE_PAD_FACTOR,
    TAG_SIZE_MM,
    ensure_odd,
    get_profile,
    select_water_profile_auto,
)
from .contours import (
    build_fish_color_mask,
    contour_aspect_ratio,
    contour_extent,
    contour_mean_hsv,
    contour_overlap_ratio_with_mask,
    contour_solidity,
    contour_touches_border,
    detect_adaptive_body_contours,
    detect_color_object_contours,
    detect_object_contours,
    preprocess_gray,
)


def major_axis_from_points(points_xy: np.ndarray):
    if points_xy.shape[0] < 5:
        return None

    pts = points_xy.astype(np.float32)
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-6:
        return None
    axis = axis / axis_norm

    projections = centered @ axis
    lo = float(np.percentile(projections, 2.0))
    hi = float(np.percentile(projections, 98.0))
    if hi <= lo:
        return None

    p1 = center + (lo * axis)
    p2 = center + (hi * axis)
    length_mm = float(np.linalg.norm(p2 - p1))
    return length_mm, p1.astype(np.float32), p2.astype(np.float32)


def _extract_measurements(
    original_image: np.ndarray,
    contours: list,
    ignore_mask: np.ndarray,
    fish_color_mask: np.ndarray,
    H,
    H_inv,
    w: int,
    h: int,
    tag_side_px: float,
    tag_area_px: float,
    max_area_px: float,
    bg_area_px: float,
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
    object_min_saturation: float,
    object_min_value: float,
    min_extent: float,
    min_fishness: float,
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
        axis_mm = major_axis_from_points(cnt_mm)
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
    local_measurements = [m for m in local_measurements if m[4] >= min_fishness]
    return [(m[0], m[1], m[2], m[3]) for m in local_measurements]


def _score_analysis_result(result: dict) -> tuple[float, int, float]:
    lengths = result.get("lengths_mm", [])
    if not result.get("tag_detected"):
        return (-1.0, -1, -1.0)
    if not lengths:
        return (0.0, 0, 0.0)
    max_len = float(max(lengths))
    total_len = float(sum(lengths))
    return (max_len, int(result.get("fish_count", 0)), total_len)


def _analyze_with_profile(
    original_image: np.ndarray,
    image_path: str,
    profile_name: str,
) -> dict:
    result = {
        "image_path": image_path,
        "water_profile": profile_name,
        "tag_detected": False,
        "segmentation_mode": "none",
        "detected_contours": 0,
        "fish_count": 0,
        "lengths_mm": [],
        "continue_processing": True,
        "overlay_path": None,
        "overlay_image": original_image.copy(),
        "tag_id": None,
        "tag_corners": None,
        "measurements": [],
    }

    profile = get_profile(profile_name)
    h, w = original_image.shape[:2]
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
        return result

    corners = det.corners.astype(np.float32) / float(det_scale)
    result["tag_detected"] = True
    result["tag_id"] = int(det.tag_id)
    result["tag_corners"] = corners

    dst_mm = np.array(
        [[0, 0], [TAG_SIZE_MM, 0], [TAG_SIZE_MM, TAG_SIZE_MM], [0, TAG_SIZE_MM]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst_mm)
    H_inv = np.linalg.inv(H)

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
            min_extent,
            0.42,
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
            0.85 * min_extent,
            0.38,
        ),
        (
            0.18 * min_area_px_dynamic,
            0.60 * min_aspect_ratio,
            1.50 * max_aspect_ratio,
            0.38 * min_perimeter_px,
            0.55 * min_solidity,
            0.55 * min_length_mm,
            max_length_mm,
            0.50 * min_length_px_factor,
            0.45 * min_thickness_px_factor,
            0.0,
            0.70 * min_extent,
            0.30,
        ),
    ]

    chosen_mode = "none"
    chosen_contour_count = 0
    measurements = []
    best_score = (-1.0, -1, -1.0)
    for pass_index, params in enumerate(pass_params):
        for mode_name, contours, mode_overlap_min in contour_sets:
            params_with_mode = params[:9] + (max(params[9], mode_overlap_min),) + params[10:]
            local_measurements = _extract_measurements(
                original_image,
                contours,
                ignore_mask,
                fish_color_mask,
                H,
                H_inv,
                w,
                h,
                tag_side_px,
                tag_area_px,
                max_area_px,
                bg_area_px,
                *params_with_mode[:10],
                object_min_saturation,
                object_min_value,
                *params_with_mode[10:],
            )
            if local_measurements:
                max_len = float(local_measurements[0][0])
                total_len = float(sum(item[0] for item in local_measurements))
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

    if measurements:
        contour_areas = [cv2.contourArea(item[1]) for item in measurements]
        max_area = max(contour_areas) if contour_areas else 0.0
        if max_area > 0.0:
            measurements = [item for item in measurements if cv2.contourArea(item[1]) >= (0.35 * max_area)]
            measurements.sort(key=lambda x: x[0], reverse=True)

    result["segmentation_mode"] = chosen_mode
    result["detected_contours"] = chosen_contour_count
    result["fish_count"] = len(measurements)
    result["lengths_mm"] = [float(item[0]) for item in measurements]
    result["measurements"] = measurements

    overlay = original_image.copy()
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

    result["overlay_image"] = overlay
    return result


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

    h, w = original_image.shape[:2]
    disp_scale = min(MAX_WIDTH / w, MAX_HEIGHT / h, 1.0)
    disp_w, disp_h = int(w * disp_scale), int(h * disp_scale)

    selected_profile_name = select_water_profile_auto(original_image) if water_profile == "auto" else water_profile
    candidate_profile_names = (
        [selected_profile_name, "clear", "murky"]
        if water_profile == "auto"
        else [water_profile]
    )
    seen_profiles = []
    for profile_name in candidate_profile_names:
        if profile_name not in seen_profiles:
            seen_profiles.append(profile_name)

    candidate_results = [_analyze_with_profile(original_image, image_path, profile_name) for profile_name in seen_profiles]

    if water_profile != "auto":
        best_result = max(candidate_results, key=_score_analysis_result)
    else:
        auto_result = next(result for result in candidate_results if result["water_profile"] == selected_profile_name)
        actual_tag_results = [
            result
            for result in candidate_results
            if result.get("tag_detected") and int(result.get("tag_id", -1) or -1) >= 0
        ]

        if auto_result["lengths_mm"] and int(auto_result.get("tag_id", -1) or -1) >= 0:
            best_result = auto_result
        elif actual_tag_results:
            best_result = max(actual_tag_results, key=_score_analysis_result)
        else:
            best_result = max(candidate_results, key=_score_analysis_result)
    result.update(
        {
            "water_profile": best_result["water_profile"],
            "tag_detected": best_result["tag_detected"],
            "segmentation_mode": best_result["segmentation_mode"],
            "detected_contours": best_result["detected_contours"],
            "fish_count": best_result["fish_count"],
            "lengths_mm": best_result["lengths_mm"],
        }
    )

    overlay = best_result["overlay_image"] if best_result["tag_detected"] else original_image.copy()
    if not best_result["tag_detected"]:
        if save_overlay_dir:
            out_dir = Path(save_overlay_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(image_path).stem}_measurement.jpg"
            no_tag_overlay = overlay.copy()
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
