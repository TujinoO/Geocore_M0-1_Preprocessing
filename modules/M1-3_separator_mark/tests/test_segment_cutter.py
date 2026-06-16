import tempfile
import unittest
from pathlib import Path

import numpy as np

from geocore_m1_3.config import SegmentationConfig
from geocore_m1_3.depth.mapper import DepthMapper
from geocore_m1_3.segments.cutter import cut_core_segments


class SegmentCutterTest(unittest.TestCase):
    def test_cuts_ten_centimeter_segments(self):
        strip_rgba = np.zeros((100, 20, 4), dtype=np.uint8)
        strip_rgba[:, :, :3] = 100
        strip_rgba[:, :, 3] = 255
        strip_mask = np.ones((100, 20), dtype=bool)
        mapper = DepthMapper(depth_start_m=0.0, depth_end_m=1.0, strip_height_px=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            segments = cut_core_segments(
                strip_rgba=strip_rgba,
                strip_mask=strip_mask,
                mapper=mapper,
                output_dir=tmpdir,
                hole_id="DH001",
                core_box_id="BOX001",
                config=SegmentationConfig(segment_length_cm=10.0, overlap_cm=0.0),
            )
            self.assertEqual(len(segments), 10)
            self.assertTrue(Path(segments[0].image_path).exists())
            self.assertAlmostEqual(segments[0].depth_start_m, 0.0)
            self.assertAlmostEqual(segments[-1].depth_end_m, 1.0)


if __name__ == "__main__":
    unittest.main()
