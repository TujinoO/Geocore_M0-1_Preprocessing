import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from geocore_m1_3.pipeline import _physical_slot_warnings, run_segment_depth


class PipelineTest(unittest.TestCase):
    def test_merged_mask_fails_closed_with_reviewable_rgb_proposal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.full((500, 300, 3), 150, dtype=np.uint8)
            for x in (20, 72, 124, 176, 228, 280):
                image[30:470, x - 4:x + 5] = (20, 20, 20)
            image_path = root / "box.png"
            mask_dir = root / "mask"
            mask_dir.mkdir()
            Image.fromarray(image, mode="RGB").save(image_path)
            Image.fromarray(np.full((500, 300), 255, dtype=np.uint8), mode="L").save(mask_dir / "mask.png")
            with self.assertRaisesRegex(ValueError, "physical-slot review proposal"):
                run_segment_depth({
                    "image_path": str(image_path), "m1_2_output_dir": str(mask_dir),
                    "output_dir": str(root / "out"), "task_id": "merged",
                    "depth_start_m": 0.0, "depth_end_m": 1.0,
                })
            preflight = json.loads((root / "out" / "merged" / "lane_preflight_review.json").read_text())
            self.assertEqual(preflight["rgb_dark_rail_candidate"]["count"], 5)
            self.assertEqual(len(preflight["suggested_lane_dividers_x"]), 4)
            self.assertFalse(preflight["suggestion_is_reviewed"])
            self.assertTrue((root / "out" / "merged" / "rgb_dark_rail_suggestion.jpg").exists())

    def test_source_prior_disagreement_is_review_only(self):
        prior = {"physical_slot_count": 5, "support_sample_ids": ["sample_014", "sample_015"]}
        warnings = _physical_slot_warnings(prior, 3, reviewed_dividers=False)
        self.assertEqual([item["code"] for item in warnings],
                         ["source_slot_count_prior_disagreement", "physical_slot_count_unverified"])
        self.assertEqual(_physical_slot_warnings(prior, 5, reviewed_dividers=True), [])

    def test_reviewed_five_slots_include_an_empty_slot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = np.full((180, 240, 3), 20, dtype=np.uint8)
            mask = np.zeros((180, 240), dtype=np.uint8)
            for left in (30, 70, 150, 190):
                image[15:165, left:left + 22] = (120, 130, 125)
                mask[15:165, left:left + 22] = 255
            image_path = root / "box.png"
            mask_dir = root / "mask"
            mask_dir.mkdir()
            Image.fromarray(image, mode="RGB").save(image_path)
            Image.fromarray(mask, mode="L").save(mask_dir / "mask.png")
            result = run_segment_depth({
                "image_path": str(image_path), "m1_2_output_dir": str(mask_dir),
                "output_dir": str(root / "out"), "hole_id": "H",
                "core_box_id": "BOX", "depth_start_m": 0.0, "depth_end_m": 1.0,
                "layout": {"lane_dividers_x": [60, 100, 140, 180]},
            })
            self.assertEqual(result["metrics"]["lane_count_detected"], 5)
            self.assertIn("empty_physical_slots", [item["code"] for item in result["warnings"]])

    def test_pipeline_runs_on_synthetic_m1_2_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "box.png"
            m12_dir = root / "m1_2"
            output_dir = root / "out"
            m12_dir.mkdir()

            image = np.zeros((180, 240, 3), dtype=np.uint8)
            image[:, :] = [20, 20, 20]
            mask = np.zeros((180, 240), dtype=bool)
            for x0 in [30, 70, 110, 150, 190]:
                image[15:165, x0 : x0 + 24] = [120, 130, 125]
                mask[15:165, x0 : x0 + 24] = True
            mask[5:20, 2:10] = True
            Image.fromarray(image, mode="RGB").save(image_path)
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(m12_dir / "mask.png")
            (m12_dir / "metadata.json").write_text(
                '{"image_width": 240, "image_height": 180, "input_path": "' + str(image_path).replace("\\", "\\\\") + '"}',
                encoding="utf-8",
            )

            result = run_segment_depth(
                {
                    "image_path": str(image_path),
                    "m1_2_output_dir": str(m12_dir),
                    "output_dir": str(output_dir),
                    "hole_id": "DH001",
                    "core_box_id": "BOX001",
                    "depth_start_m": 0.0,
                    "depth_end_m": 1.0,
                    "task_id": "test_task",
                    "segmentation": {"segment_length_cm": 20.0},
                }
            )
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["metrics"]["lane_count_detected"], 5)
            self.assertEqual(result["metrics"]["segment_count"], 5)
            self.assertTrue((output_dir / "test_task" / "reconstructed_strip.png").exists())
            self.assertTrue((output_dir / "test_task" / "segments.json").exists())


if __name__ == "__main__":
    unittest.main()
