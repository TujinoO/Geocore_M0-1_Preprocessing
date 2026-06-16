import unittest

import numpy as np

from geocore_m1_3.config import LayoutConfig
from geocore_m1_3.layout.lane_detector import detect_lanes


class LaneDetectorTest(unittest.TestCase):
    def test_detects_five_lanes_with_small_outside_noise(self):
        mask = np.zeros((160, 220), dtype=bool)
        for i, x0 in enumerate([25, 60, 95, 130, 165]):
            mask[10:150, x0 : x0 + 18] = True
            if i % 2 == 0:
                mask[55:60, x0 : x0 + 18] = False
        mask[20:40, 2:8] = True
        lanes = detect_lanes(mask, LayoutConfig(lane_count=5, lane_padding_px=2), core_box_id="BOX")
        self.assertEqual(len(lanes), 5)
        centers = [(lane.bbox[0] + lane.bbox[2]) / 2 for lane in lanes]
        self.assertTrue(all(left < right for left, right in zip(centers, centers[1:])))
        self.assertGreater(lanes[0].bbox[0], 5)


if __name__ == "__main__":
    unittest.main()
