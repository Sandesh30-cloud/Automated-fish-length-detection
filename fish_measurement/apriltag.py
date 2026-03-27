import cv2
import numpy as np
from dataclasses import dataclass

from .config import APRILTAG_DETECTORS


@dataclass
class TagDetection:
    corners: np.ndarray
    tag_id: int = -1
    decision_margin: float = 0.0
    hamming: int = 0


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected four 2D points")

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _tag_geometry_metrics(corners: np.ndarray) -> tuple[float, float, float]:
    sides = np.array(
        [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)],
        dtype=np.float32,
    )
    side_px = float(np.mean(sides))
    if side_px <= 1e-6:
        return 0.0, 0.0, float("inf")

    area = float(abs(cv2.contourArea(corners.astype(np.float32))))
    area_ratio = area / max(1e-6, side_px * side_px)
    side_ratio = float(np.max(sides) / max(1e-6, np.min(sides)))
    return side_px, area_ratio, side_ratio


def _tag_geometry_score(corners: np.ndarray) -> float:
    _, area_ratio, side_ratio = _tag_geometry_metrics(corners)
    area_term = max(0.0, 1.0 - (abs(area_ratio - 1.0) / 1.2))
    side_term = max(0.0, 1.0 - ((side_ratio - 1.0) / 4.0))
    return (0.65 * area_term) + (0.35 * side_term)


def _detect_tag_quad_fallback(gray: np.ndarray, image_shape: tuple, min_side_px: float) -> TagDetection | None:
    height, width = image_shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4
    )
    mask = cv2.bitwise_or(edges, cv2.bitwise_not(binary))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_detection = None
    best_score = -1e9
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < max(500.0, 0.6 * min_side_px * min_side_px):
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        corners = _order_quad_points(approx.reshape(4, 2))
        side_px, area_ratio, side_ratio = _tag_geometry_metrics(corners)
        x, y, w_box, h_box = cv2.boundingRect(corners.astype(np.int32))
        touches_border = x <= 1 or y <= 1 or (x + w_box) >= (width - 1) or (y + h_box) >= (height - 1)
        if side_px < min_side_px or area_ratio < 0.45 or side_ratio > 2.8 or touches_border:
            continue

        dst = np.array([[0, 0], [119, 0], [119, 119], [0, 119]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        patch = cv2.warpPerspective(gray, matrix, (120, 120))
        if patch.size == 0:
            continue

        border = np.concatenate(
            [
                patch[:18, :].ravel(),
                patch[-18:, :].ravel(),
                patch[:, :18].ravel(),
                patch[:, -18:].ravel(),
            ]
        )
        inner = patch[26:94, 26:94]
        if inner.size == 0:
            continue

        border_mean = float(np.mean(border))
        inner_mean = float(np.mean(inner))
        contrast = border_mean - inner_mean
        if contrast < 18.0:
            continue

        center = np.mean(corners, axis=0)
        left_bonus = max(0.0, 1.0 - (float(center[0]) / max(1.0, 0.55 * width)))
        top_bonus = max(0.0, 1.0 - (float(center[1]) / max(1.0, 0.55 * height)))
        geometry_score = _tag_geometry_score(corners)
        score = contrast + (12.0 * geometry_score) + (0.025 * side_px) + (4.0 * left_bonus) + (2.0 * top_bonus)

        if score > best_score:
            best_score = score
            best_detection = TagDetection(corners=corners, tag_id=-1, decision_margin=contrast, hamming=0)

    return best_detection


def detect_apriltag(gray_candidates: list, image_shape: tuple, profile: dict) -> tuple:
    height, width = image_shape[:2]
    x_limit = float(profile["tag_expected_x_max_ratio"]) * width
    y_limit = float(profile["tag_expected_y_min_ratio"]) * height
    location_weight = float(profile["tag_location_weight"])
    min_side_px = float(profile["tag_min_side_px"])

    def search_candidates(candidate_subset: list, include_otsu: bool) -> tuple:
        best_detection = None
        best_scale = 1.0
        best_score = -1e9
        best_in_region_detection = None
        best_in_region_scale = 1.0
        best_in_region_score = -1e9

        for gray, scale in candidate_subset:
            thresholded = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
            )
            detection_inputs = [gray, thresholded, cv2.bitwise_not(thresholded)]
            if include_otsu:
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                detection_inputs.append(otsu)

            for image in detection_inputs:
                for detector in APRILTAG_DETECTORS:
                    detections = detector.detect(image)
                    for det in detections:
                        corners = det.corners.astype(np.float32) / float(scale)
                        side_px, area_ratio, side_ratio = _tag_geometry_metrics(corners)
                        if side_px < min_side_px:
                            continue
                        center = np.mean(corners, axis=0)
                        in_expected_region = center[0] <= x_limit and center[1] >= y_limit

                        hamming = int(getattr(det, "hamming", 99))
                        decision_margin = float(getattr(det, "decision_margin", 0.0))
                        score = decision_margin - (20.0 * hamming) + (0.04 * side_px)
                        score += location_weight if in_expected_region else (-0.6 * location_weight)
                        score += 24.0 * _tag_geometry_score(corners)
                        score -= 80.0 * max(0.0, 0.18 - area_ratio)
                        score -= 6.0 * max(0.0, side_ratio - 3.5)

                        if score > best_score:
                            best_score = score
                            best_detection = det
                            best_scale = float(scale)
                        if (
                            in_expected_region
                            and area_ratio >= 0.45
                            and side_ratio <= 3.8
                            and score > best_in_region_score
                        ):
                            best_in_region_score = score
                            best_in_region_detection = det
                            best_in_region_scale = float(scale)

        if best_in_region_detection is not None:
            return best_in_region_detection, best_in_region_scale
        if best_detection is not None:
            corners = best_detection.corners.astype(np.float32) / float(best_scale)
            _, area_ratio, side_ratio = _tag_geometry_metrics(corners)
            if area_ratio >= 0.45 and side_ratio <= 3.8:
                return best_detection, best_scale
        return None, 1.0

    base_candidates = [(gray, scale) for gray, scale in gray_candidates if abs(float(scale) - 1.0) < 1e-6]
    scaled_candidates = [(gray, scale) for gray, scale in gray_candidates if abs(float(scale) - 1.0) >= 1e-6]

    det, scale = search_candidates(base_candidates, include_otsu=False)
    if det is not None:
        return det, scale

    if scaled_candidates:
        det, scale = search_candidates(scaled_candidates, include_otsu=False)
        if det is not None:
            return det, scale

    det, scale = search_candidates(gray_candidates, include_otsu=True)
    if det is not None:
        return det, scale

    for gray, _ in gray_candidates:
        fallback_detection = _detect_tag_quad_fallback(gray, image_shape, min_side_px)
        if fallback_detection is not None:
            return fallback_detection, 1.0

    return None, 1.0
