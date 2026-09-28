import unittest

import numpy as np

from geocore_m1_3.layout.rgb_slot_geometry import estimate_dark_rail_geometry, estimate_rgb_slot_geometry


class RgbSlotGeometryTest(unittest.TestCase):
    def test_dark_rails_recover_five_physical_slots_from_merged_mask_scene(self):
        rgb = np.full((500, 300, 3), 150, dtype=np.uint8)
        for x in (20, 72, 124, 176, 228, 280):
            rgb[30:470, x - 4:x + 5] = (20, 20, 20)
        candidate = estimate_dark_rail_geometry(rgb)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.count, 5)
        self.assertFalse(candidate.review_required)
        self.assertEqual(len(candidate.separator_x), 6)

    def test_central_dark_bands_without_outer_rails_require_review(self):
        rgb = np.full((500, 300, 3), 150, dtype=np.uint8)
        for x in (65, 105, 145, 185, 225):
            rgb[30:470, x - 4:x + 5] = (20, 20, 20)
        candidate = estimate_dark_rail_geometry(rgb)
        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.review_required)
        self.assertIn("outer_rail_missing_or_box_crop_incomplete", candidate.review_reasons)

    def test_coloured_physical_dividers_with_empty_slot(self):
        for count in (2, 5, 6):
            with self.subTest(count=count):
                width = count * 70 + 40
                rgb = np.full((480, width, 3), 18, dtype=np.uint8)
                mask = np.zeros((480, width), dtype=bool)
                for index in range(count + 1):
                    x = 20 + index * 70
                    rgb[20:460, x - 3:x + 4] = (220, 85, 12)
                for index in range(count):
                    if index == 2:
                        continue  # a physical slot can contain no core
                    x = 20 + index * 70
                    rgb[40:440, x + 9:x + 61] = (115, 108, 95)
                    mask[40:440, x + 9:x + 61] = True
                candidate = estimate_rgb_slot_geometry(rgb, mask)
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate.count, count)


if __name__ == "__main__":
    unittest.main()
