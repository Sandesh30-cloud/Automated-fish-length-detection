import os
import argparse
from fish_measurement.analysis import analyze_image, process_directory_loop, process_image


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
