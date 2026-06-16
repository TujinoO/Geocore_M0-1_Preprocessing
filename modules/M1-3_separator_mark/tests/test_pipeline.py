import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from geocore_m1_3.pipeline import run_segment_depth


class PipelineTest(unittest.TestCase):
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
