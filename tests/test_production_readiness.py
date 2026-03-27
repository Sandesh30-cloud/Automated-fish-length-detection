import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from automation_controller import (
    get_archive_target_path,
    parse_extensions_csv as parse_controller_extensions,
    validate_runtime_args,
)
from fish_measurement.analysis import analyze_image
from homography import parse_extensions_csv as parse_homography_extensions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "processed"


class ExtensionParsingTests(unittest.TestCase):
    def test_extension_parsing_normalizes_values(self) -> None:
        self.assertEqual(
            parse_controller_extensions("jpg, .PNG ,jpeg"),
            (".jpg", ".png", ".jpeg"),
        )
        self.assertEqual(
            parse_homography_extensions("jpg, .PNG ,jpeg"),
            (".jpg", ".png", ".jpeg"),
        )

    def test_extension_parsing_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            parse_controller_extensions(" , ")


class RuntimeValidationTests(unittest.TestCase):
    def test_validate_runtime_args_rejects_overlapping_directories(self) -> None:
        shared = PROJECT_ROOT / "images"
        args = Namespace(
            settle_seconds=0.5,
            interval_seconds=2.0,
            fish_threshold=0.5,
            fish_margin=0.15,
            max_iterations=None,
            source="directory",
            archive_dir=None,
            input_dir=shared,
            results_dir=shared,
        )
        with self.assertRaises(ValueError):
            validate_runtime_args(args)

    def test_archive_target_gets_unique_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_dir = Path(tmp_dir)
            existing = archive_dir / "sample.jpg"
            existing.write_bytes(b"existing")
            candidate = get_archive_target_path(archive_dir, existing)
            self.assertEqual(candidate.name, "sample_0001.jpg")


class MeasurementSmokeTests(unittest.TestCase):
    def test_known_problem_images_now_measure(self) -> None:
        expected = {
            "image5.jpeg": 1,
            "image7.jpeg": 1,
            "image9.jpeg": 1,
            "image12.jpeg": 1,
        }
        for image_name, min_count in expected.items():
            with self.subTest(image=image_name):
                result = analyze_image(
                    str(SAMPLE_DIR / image_name),
                    visualize=False,
                )
                self.assertTrue(result["tag_detected"], image_name)
                self.assertGreaterEqual(result["fish_count"], min_count, image_name)
                self.assertTrue(result["lengths_mm"], image_name)


if __name__ == "__main__":
    unittest.main()
