import unittest

from geocore_m1_1.review import _map_review_points_to_raw


class ReviewCoordinatesTest(unittest.TestCase):
    def test_downsampled_review_rectangle_maps_to_full_resolution(self):
        raw_bbox, clipped, _ = _map_review_points_to_raw(
            [(0, 0), (499, 999)],
            [100, 200, 1099, 2199],
            [500, 1000],
            [999 / 499, 1999 / 999],
        )
        self.assertEqual(raw_bbox, (100, 200, 1099, 2199))
        self.assertEqual(clipped, [(0, 0), (499, 999)])

    def test_legacy_full_resolution_metadata_still_maps_one_to_one(self):
        raw_bbox, _, _ = _map_review_points_to_raw(
            [(20, 30), (80, 100)], [100, 200, 299, 399], None, None
        )
        self.assertEqual(raw_bbox, (120, 230, 180, 300))
