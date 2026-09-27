import unittest

import numpy as np

from geocore_m1_3.config import ReconstructionConfig
from geocore_m1_3.layout.lane_detector import Lane
from geocore_m1_3.reconstruct.lane_strip import build_lane_strip


class GapPolicyTest(unittest.TestCase):
    def test_default_preserves_large_gap_and_explicit_close_warns(self):
        image = np.zeros((100, 20, 3), dtype=np.uint8)
        mask = np.zeros((100, 20), dtype=bool)
        mask[:30, 5:15] = True
        mask[70:, 5:15] = True
        lane = Lane("L1", 1, 1, [0, 0, 20, 100], [[10, 0], [10, 100]],
                    "top_to_bottom", 100, int(mask.sum()), 1.0)
        default = ReconstructionConfig(large_gap_warning_px=20)
        preserved = build_lane_strip(image, mask, lane, 20, default)
        self.assertEqual(default.gap_policy, "preserve_all_gaps")
        self.assertEqual(preserved.image_rgba.shape[0], 100)
        self.assertEqual(preserved.warnings[0]["code"], "large_gap_preserved_requires_review")
        compressed = build_lane_strip(
            image, mask, lane, 20,
            ReconstructionConfig(gap_policy="close_artificial_gaps", large_gap_warning_px=20),
        )
        self.assertLess(compressed.image_rgba.shape[0], 100)
        self.assertEqual(compressed.warnings[0]["code"], "large_gap_requires_review")


if __name__ == "__main__":
    unittest.main()
