from __future__ import annotations

import unittest
from pathlib import Path

from geocore_m1_1.config import M11Config
from geocore_m1_1.io_envi import read_envi_image
from geocore_m1_1.segment_boxes import detect_core_boxes


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "assets" / "legacy_rgb_20230909"


class SamplePipelineTests(unittest.TestCase):
    def test_sample_detection_finds_nine_boxes(self) -> None:
        image, _ = read_envi_image(
            SAMPLE_ROOT / "RGB-20230909_141858-00000.dat",
            SAMPLE_ROOT / "RGB-20230909_141858-00000.hdr",
        )
        detection = detect_core_boxes(image, M11Config(thumbnail_width=512))
        self.assertEqual(len(detection.candidates), 9)


if __name__ == "__main__":
    unittest.main()
