from __future__ import annotations

import unittest

import numpy as np

from geocore_m1_1.config import M11Config
from geocore_m1_1.geometry import estimate_box_angle
from geocore_m1_1.image_ops import rotate_nearest


class RotationTests(unittest.TestCase):
    def test_orange_frame_angle_tracks_frame_not_rock_texture(self) -> None:
        image = np.zeros((900, 300, 3), dtype=np.uint8)
        for x in (20, 100, 200, 280):
            image[:, x:x + 8] = (230, 90, 12)
        image[:30] = (230, 90, 12)
        image[-30:] = (230, 90, 12)
        tilted, _ = rotate_nearest(image, 1.0)
        angle = estimate_box_angle(
            tilted, (0, 0, tilted.shape[1] - 1, tilted.shape[0] - 1), M11Config()
        )
        self.assertEqual(angle.method, "orange_vertical_frame")
        self.assertAlmostEqual(angle.angle_deg, -1.0, delta=0.12)

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
