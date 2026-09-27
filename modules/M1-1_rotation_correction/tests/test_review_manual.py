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
            review_w, review_h = target.review_source_size_px or (0, 0)
            annotation = {
                "type": "rectangle",
                "rect": {
                    "x": 40,
                    "y": 40,
                    "width": max(20, review_w - 80),
                    "height": max(20, review_h - 80),
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
            self.assertGreater(updated["bbox_xyxy_raw"][2] - updated["bbox_xyxy_raw"][0], review_w)


if __name__ == "__main__":
    unittest.main()
