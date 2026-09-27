import unittest

import numpy as np

from geocore_m1_3.config import LayoutConfig
from geocore_m1_3.layout.lane_detector import detect_lanes, estimate_lane_count


class LaneDetectorTest(unittest.TestCase):
    def test_auto_three_lanes_and_rejects_wrong_explicit_count(self):
        mask = np.zeros((300, 240), dtype=bool)
        for x0 in (18, 92, 166):
            mask[12:287, x0:x0 + 52] = True
        estimate = estimate_lane_count(mask)
        self.assertEqual(estimate.count, 3)
        self.assertEqual(estimate.reason, "ok")
        lanes = detect_lanes(mask, LayoutConfig(), core_box_id="ORANGE")
        self.assertEqual(len(lanes), 3)
        self.assertTrue(all(a.bbox[2] <= b.bbox[0] for a, b in zip(lanes, lanes[1:])))
        with self.assertRaisesRegex(ValueError, "conflicts"):
            detect_lanes(mask, LayoutConfig(lane_count=5))

    def test_auto_fails_closed_when_lanes_are_ambiguous(self):
        mask = np.zeros((300, 240), dtype=bool)
        mask[20:280, 15:225] = True
        estimate = estimate_lane_count(mask)
        self.assertNotEqual(estimate.reason, "ok")
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            detect_lanes(mask, LayoutConfig())

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
