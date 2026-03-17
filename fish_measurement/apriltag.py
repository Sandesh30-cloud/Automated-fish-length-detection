import cv2
import numpy as np

from .config import APRILTAG_DETECTORS


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

