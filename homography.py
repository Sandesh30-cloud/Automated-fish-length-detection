import argparse
import logging
import sys
from pathlib import Path

from fish_measurement.analysis import analyze_image, process_directory_loop
from fish_measurement.config import WATER_PROFILES


LOGGER = logging.getLogger("homography")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_extensions_csv(extensions: str) -> tuple[str, ...]:
    parsed = tuple(
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in extensions.split(",")
        if ext.strip()
    )
    if not parsed:
        raise ValueError("No valid extensions provided.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Homography-based fish measurement")
    parser.add_argument("--image", type=Path, default=None, help="Single image path")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically process new images from a directory loop",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("images"),
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show OpenCV visualization window.",
    )
    parser.add_argument(
        "--save-overlay-dir",
        type=Path,
        default=None,
        help="Optional directory where rendered measurement overlays will be written.",
    )
    parser.add_argument(
        "--water-profile",
        choices=["auto"] + sorted(WATER_PROFILES.keys()),
        default="auto",
        help="Segmentation profile tuned for water conditions.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be > 0")
    if args.auto and args.image is not None:
        raise ValueError("Use either --image or --auto, not both.")
    if not args.auto and args.image is None:
        raise ValueError("Provide --image for single-image mode or use --auto.")
    if not args.auto and not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    return parse_extensions_csv(args.extensions)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        extensions = validate_args(args)

        if args.auto:
            process_directory_loop(
                input_dir=str(args.input_dir),
                interval_seconds=args.interval_seconds,
                extensions=extensions,
                visualize=args.visualize,
                water_profile=args.water_profile,
            )
            return 0

        analysis = analyze_image(
            str(args.image),
            visualize=args.visualize,
            wait_for_key=args.visualize,
            water_profile=args.water_profile,
            save_overlay_dir=str(args.save_overlay_dir) if args.save_overlay_dir else None,
        )
        if not args.visualize:
            if not analysis["tag_detected"]:
                LOGGER.warning("No AprilTag detected in %s", args.image)
            else:
                LOGGER.info(
                    "Measured %d fish from %s using %s profile: %s",
                    analysis["fish_count"],
                    args.image,
                    analysis["water_profile"],
                    ", ".join(f"{length:.2f} mm" for length in analysis["lengths_mm"]) or "no lengths",
                )
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
        return 0
    except Exception:
        LOGGER.exception("Homography CLI failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
