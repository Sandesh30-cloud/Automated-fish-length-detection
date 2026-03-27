import argparse
import csv
import importlib.util
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Optional, Set, Tuple

import cv2

import homography


LOGGER = logging.getLogger("automation_controller")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_extensions_csv(extensions: str) -> Tuple[str, ...]:
    parsed = tuple(
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in extensions.split(",")
        if ext.strip()
    )
    if not parsed:
        raise ValueError("No valid image extensions provided in --extensions")
    return parsed


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.settle_seconds < 0:
        raise ValueError("--settle-seconds must be >= 0")
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be > 0")
    if not 0.0 <= args.fish_threshold <= 1.0:
        raise ValueError("--fish-threshold must be between 0 and 1")
    if not 0.0 <= args.fish_margin <= 1.0:
        raise ValueError("--fish-margin must be between 0 and 1")
    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max-iterations must be > 0 when provided")
    if args.source == "camera" and args.archive_dir is not None:
        raise ValueError("--archive-dir is only supported with --source directory")
    if args.source == "directory" and args.archive_dir is not None:
        if args.archive_dir.resolve() == args.input_dir.resolve():
            raise ValueError("--archive-dir must be different from --input-dir")
    if args.results_dir.resolve() == args.input_dir.resolve():
        raise ValueError("--results-dir must be different from --input-dir")


def load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_image(camera: cv2.VideoCapture, output_dir: Path) -> Path:
    ok, frame = camera.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to capture image from camera")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = output_dir / f"capture_{timestamp}.jpg"

    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Failed to write captured image to {image_path}")
    return image_path


def get_next_image_from_directory(
    input_dir: Path,
    allowed_extensions: Tuple[str, ...],
    settle_seconds: float,
) -> Optional[Path]:
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        return None

    now = time.time()
    candidates = sorted(input_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    for image_path in candidates:
        if not image_path.is_file():
            continue
        if image_path.name.startswith(".") or image_path.name.startswith("._"):
            continue
        if image_path.suffix.lower() not in allowed_extensions:
            continue
        if (now - image_path.stat().st_mtime) < settle_seconds:
            continue
        return image_path
    return None


def get_archive_target_path(archive_dir: Path, image_path: Path) -> Path:
    target = archive_dir / image_path.name
    if not target.exists():
        return target

    stem = image_path.stem
    suffix = image_path.suffix
    for index in range(1, 10_000):
        candidate = archive_dir / f"{stem}_{index:04d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free archive filename for {image_path.name}")


def process_iteration(
    image_path: Path,
    inference_module: ModuleType,
    fish_model,
    fish_threshold: float,
    fish_margin: float,
    fish_positive_when: str,
    results_dir: Path,
    visualize: bool,
) -> None:
    LOGGER.info("Processing image: %s", image_path)

    prediction = robust_fish_prediction(
        image_path=image_path,
        inference_module=inference_module,
        fish_model=fish_model,
        fish_threshold=fish_threshold,
        fish_margin=fish_margin,
        fish_positive_when=fish_positive_when,
    )
    if visualize:
        show_fish_detection_window(image_path=image_path, prediction=prediction)

    if not prediction["fish_present"]:
        LOGGER.info(
            "No fish detected (confidence=%.3f, raw_probability=%.3f).",
            prediction["confidence"],
            prediction["raw_probability"],
        )
        save_result_record(
            results_dir=results_dir,
            image_path=image_path,
            prediction=prediction,
            analysis=None,
        )
        return

    LOGGER.info(
        "Fish detected (confidence=%.3f, raw_probability=%.3f). Running homography analysis.",
        prediction["confidence"],
        prediction["raw_probability"],
    )

    analysis = homography.analyze_image(
        str(image_path),
        visualize=visualize,
        wait_for_key=False,
        save_overlay_dir=str(results_dir),
    )
    fish_count = analysis["fish_count"]
    lengths = analysis["lengths_mm"]
    if fish_count == 0:
        LOGGER.info("Fish count: 0 | lengths_mm: []")
        save_result_record(
            results_dir=results_dir,
            image_path=image_path,
            prediction=prediction,
            analysis=analysis,
        )
        return

    lengths_str = ", ".join(f"{length:.2f}" for length in lengths)
    LOGGER.info("Fish count: %d | lengths_mm: [%s]", fish_count, lengths_str)
    save_result_record(
        results_dir=results_dir,
        image_path=image_path,
        prediction=prediction,
        analysis=analysis,
    )


def robust_fish_prediction(
    image_path: Path,
    inference_module: ModuleType,
    fish_model,
    fish_threshold: float,
    fish_margin: float,
    fish_positive_when: str,
) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 7, 7, 7, 21)

    predictions = []
    for variant_name, variant_img in (
        ("original", frame),
        ("clahe", clahe_bgr),
        ("denoised", denoised),
    ):
        pred = inference_module.predict_fish_from_bgr(
            img=variant_img,
            model=fish_model,
            threshold=fish_threshold,
            fish_positive_when=fish_positive_when,
        )
        pred["variant"] = variant_name
        predictions.append(pred)

    if fish_positive_when == "higher":
        best = max(predictions, key=lambda item: item["raw_probability"])
        decision_threshold = max(0.0, fish_threshold - fish_margin)
        fish_present = best["raw_probability"] > decision_threshold
        fish_confidence = float(best["raw_probability"])
    else:
        best = min(predictions, key=lambda item: item["raw_probability"])
        decision_threshold = min(1.0, fish_threshold + fish_margin)
        fish_present = best["raw_probability"] < decision_threshold
        fish_confidence = float(1 - best["raw_probability"])

    return {
        "fish_present": fish_present,
        "confidence": fish_confidence if fish_present else float(1 - fish_confidence),
        "raw_probability": float(best["raw_probability"]),
        "variant": best["variant"],
        "decision_threshold": float(decision_threshold),
        "fish_positive_when": fish_positive_when,
    }


def show_fish_detection_window(image_path: Path, prediction: dict) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        LOGGER.warning("Could not open image for visualization: %s", image_path)
        return

    label = "Fish detected" if prediction["fish_present"] else "No fish"
    confidence = prediction["confidence"]
    raw_prob = prediction["raw_probability"]
    variant = prediction.get("variant", "original")
    decision_threshold = prediction.get("decision_threshold", 0.5)
    fish_positive_when = prediction.get("fish_positive_when", "higher")
    color = (0, 200, 0) if prediction["fish_present"] else (0, 0, 220)

    cv2.putText(frame, f"{label} | conf={confidence:.2f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"raw_prob={raw_prob:.3f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(
        frame,
        f"variant={variant} | th={decision_threshold:.2f} | map={fish_positive_when}",
        (12, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    cv2.imshow("Fish Detection", frame)
    cv2.waitKey(1)


def save_result_record(
    results_dir: Path,
    image_path: Path,
    prediction: dict,
    analysis: Optional[dict],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "measurements.csv"
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "image_path",
                    "fish_present",
                    "fish_confidence",
                    "raw_probability",
                    "tag_detected",
                    "fish_count",
                    "lengths_mm",
                    "segmentation_mode",
                    "overlay_path",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                str(image_path),
                bool(prediction.get("fish_present", False)),
                float(prediction.get("confidence", 0.0)),
                float(prediction.get("raw_probability", 0.0)),
                bool(analysis.get("tag_detected", False)) if analysis else False,
                int(analysis.get("fish_count", 0)) if analysis else 0,
                ";".join(f"{v:.2f}" for v in analysis.get("lengths_mm", [])) if analysis else "",
                analysis.get("segmentation_mode", "") if analysis else "",
                analysis.get("overlay_path", "") if analysis else "",
            ]
        )
        csv_file.flush()


def run_controller(
    source: str,
    camera_index: int,
    output_dir: Path,
    input_dir: Path,
    archive_dir: Optional[Path],
    allowed_extensions: Tuple[str, ...],
    settle_seconds: float,
    interval_seconds: float,
    fish_threshold: float,
    fish_margin: float,
    fish_positive_when: str,
    exit_if_no_image: bool,
    results_dir: Path,
    visualize: bool,
    max_iterations: Optional[int],
) -> None:
    inference_path = Path(__file__).resolve().parent / "fish-present-or-not" / "inference.py"
    inference_module = load_module_from_path("fish_inference_module", inference_path)

    LOGGER.info("Loading fish model (one-time initialization)...")
    fish_model = inference_module.load_fish_model()
    LOGGER.info("Fish model loaded.")

    camera = None
    processed_files: Set[Path] = set()
    if source == "camera":
        LOGGER.info("Initializing camera index %d...", camera_index)
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")
    else:
        input_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Listening for incoming images in: %s", input_dir)
        if archive_dir is not None:
            archive_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Processed images will be moved to: %s", archive_dir)

    try:
        iteration = 0
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                LOGGER.info("Reached max iterations: %d", max_iterations)
                break

            iteration += 1
            LOGGER.info("Starting iteration %d", iteration)
            try:
                image_path: Optional[Path] = None
                if source == "camera":
                    image_path = capture_image(camera, output_dir)
                    LOGGER.info("Captured image: %s", image_path)
                else:
                    image_path = get_next_image_from_directory(
                        input_dir=input_dir,
                        allowed_extensions=allowed_extensions,
                        settle_seconds=settle_seconds,
                    )
                    if image_path is None:
                        if exit_if_no_image:
                            LOGGER.info("No image found in %s. Exiting.", input_dir)
                            break
                        LOGGER.debug("No new image found in %s", input_dir)
                        time.sleep(interval_seconds)
                        continue

                process_iteration(
                    image_path=image_path,
                    inference_module=inference_module,
                    fish_model=fish_model,
                    fish_threshold=fish_threshold,
                    fish_margin=fish_margin,
                    fish_positive_when=fish_positive_when,
                    results_dir=results_dir,
                    visualize=visualize,
                )
                if source == "directory":
                    processed_files.add(image_path)
                if source == "directory" and archive_dir is not None:
                    target = get_archive_target_path(archive_dir, image_path)
                    shutil.move(str(image_path), str(target))
                    LOGGER.info("Archived processed image: %s", target)
            except Exception:
                LOGGER.exception("Iteration %d failed.", iteration)

            time.sleep(interval_seconds)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        LOGGER.info("Resources released and OpenCV windows closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fish automation controller loop.")
    parser.add_argument(
        "--source",
        choices=["directory", "camera"],
        default="directory",
        help="Image source type. Use 'directory' for Raspberry Pi image drops.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument("--input-dir", type=Path, default=Path("images"))
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png",
        help="Comma-separated list of image extensions to process.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Minimum file age before processing to avoid partial writes.",
    )
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--fish-threshold", type=float, default=0.5)
    parser.add_argument(
        "--fish-margin",
        type=float,
        default=0.15,
        help="Extra tolerance added to fish threshold for robust multi-pass detection.",
    )
    parser.add_argument(
        "--fish-positive-when",
        choices=["higher", "lower"],
        default="higher",
        help="How to interpret model output for fish class.",
    )
    parser.add_argument(
        "--exit-if-no-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit when no image is available in input directory.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show fish detection and homography output windows.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        validate_runtime_args(args)
        allowed_extensions = parse_extensions_csv(args.extensions)

        run_controller(
            source=args.source,
            camera_index=args.camera_index,
            output_dir=args.output_dir,
            input_dir=args.input_dir,
            archive_dir=args.archive_dir,
            allowed_extensions=allowed_extensions,
            settle_seconds=args.settle_seconds,
            interval_seconds=args.interval_seconds,
            fish_threshold=args.fish_threshold,
            fish_margin=args.fish_margin,
            fish_positive_when=args.fish_positive_when,
            exit_if_no_image=args.exit_if_no_image,
            results_dir=args.results_dir,
            visualize=args.visualize,
            max_iterations=args.max_iterations,
        )
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Controller terminated with an error.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
