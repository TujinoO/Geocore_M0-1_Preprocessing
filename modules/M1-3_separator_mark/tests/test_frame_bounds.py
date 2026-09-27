import unittest

import numpy as np

from geocore_m1_3.layout.frame_bounds import trim_orange_frame_rows


class FrameBoundsTest(unittest.TestCase):
    def test_orange_end_rails_are_not_core(self):
        image = np.zeros((300, 200, 3), dtype=np.uint8)
        image[5:20, :] = (230, 90, 12)
        image[280:295, :] = (230, 90, 12)
        mask = np.ones((300, 200), dtype=bool)
        result = trim_orange_frame_rows(mask, image)
        self.assertTrue(result["applied"])
        self.assertFalse(mask[10].any())
        self.assertFalse(mask[290].any())
        self.assertTrue(mask[150].all())

    def test_no_orange_frame_leaves_mask_unchanged(self):
        image = np.zeros((300, 200, 3), dtype=np.uint8)
        mask = np.ones((300, 200), dtype=bool)
        self.assertFalse(trim_orange_frame_rows(mask, image)["applied"])
        self.assertTrue(mask.all())
