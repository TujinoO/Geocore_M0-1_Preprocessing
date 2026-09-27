from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from geocore_preprocessing.cli import build_parser
from geocore_preprocessing.paths import find_m12_model_package


class V4BindingTests(unittest.TestCase):
    def test_pipeline_cli_defaults_to_model(self):
        args = build_parser().parse_args(["run", "--output-dir", "unused"])
        self.assertEqual(args.m1_2_engine, "model")

    def test_default_package_is_v4_and_explicit_missing_does_not_fall_back(self):
        package = find_m12_model_package()
        self.assertIsNotNone(package)
        self.assertEqual(package.name, "core_mask_unet_v4")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_m12_model_package(explicit=str(Path(tmp) / "missing")))


if __name__ == "__main__":
    unittest.main()
