import unittest

import numpy as np

from geocore_mask.postprocess.contour import mask_to_components


class LocalContourTest(unittest.TestCase):
    def test_components_keep_global_coordinates_at_image_edges(self):
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[0:6, 0:8] = 1
        mask[20:26, 30:39] = 1
        result = mask_to_components(mask, max_contour_points=800)
        self.assertEqual(result["component_count"], 2)
        by_bbox = {tuple(item["bbox"]): item for item in result["components"]}
        self.assertEqual(set(by_bbox), {(0, 0, 8, 6), (30, 20, 39, 26)})
        for item in result["components"]:
            self.assertTrue(item["contour"])
            self.assertTrue(all(0 <= x < 60 and 0 <= y < 40 for x, y in item["contour"]))


if __name__ == "__main__":
    unittest.main()
