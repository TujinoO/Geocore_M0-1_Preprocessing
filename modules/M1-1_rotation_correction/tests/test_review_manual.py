from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from geocore_m1_1.config import M11Config
from geocore_m1_1.pipeline import run_m11_rotation_correction
from geocore_m1_1.review import manual_correct_box


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "assets" / "legacy_rgb_20230909"


class ManualReviewTests(unittest.TestCase):
    def test_manual_rectangle_updates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_m11_rotation_correction(
                input_path=str(SAMPLE_ROOT / "RGB-20230909_141858-00000.dat"),
                hdr_path=str(SAMPLE_ROOT / "RGB-20230909_141858-00000.hdr"),
                output_dir=tmp,
                config=M11Config(save_previews=False, expected_box_count=9),
            )
            target = next(box for box in result.boxes if box.box_id == "box_0006")
            x0, y0, x1, y1 = target.source_crop_bbox_raw or (0, 0, 0, 0)
            annotation = {
                "type": "rectangle",
                "rect": {
                    "x": 40,
                    "y": 40,
                    "width": max(20, x1 - x0 - 80),
                    "height": max(20, y1 - y0 - 80),
                },
            }
            updated = manual_correct_box(
                output_dir=tmp,
                input_path=str(SAMPLE_ROOT / "RGB-20230909_141858-00000.dat"),
                hdr_path=str(SAMPLE_ROOT / "RGB-20230909_141858-00000.hdr"),
                box_id="box_0006",
                annotation=annotation,
            )
            self.assertEqual(updated["correction_status"], "manual")
            self.assertFalse(updated["needs_manual_review"])
            self.assertTrue(Path(updated["output_image"]).exists())


if __name__ == "__main__":
    unittest.main()
