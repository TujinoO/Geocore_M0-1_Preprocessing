from __future__ import annotations

import unittest

import numpy as np

from geocore_m1_1.image_ops import rotate_nearest


class RotationTests(unittest.TestCase):
    def test_nearest_rotation_preserves_existing_values(self) -> None:
        image = np.zeros((12, 8), dtype=np.uint8)
        image[3:9, 2:6] = 7
        rotated, matrix = rotate_nearest(image, 3.0, fill_value=0)
        unique = set(np.unique(rotated).tolist())
        self.assertTrue(unique.issubset({0, 7}))
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 3)


if __name__ == "__main__":
    unittest.main()

