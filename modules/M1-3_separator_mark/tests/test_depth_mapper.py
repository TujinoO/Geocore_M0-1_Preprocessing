import unittest

from geocore_m1_3.depth.mapper import DepthMapper, MissingInterval


class DepthMapperTest(unittest.TestCase):
    def test_linear_mapping(self):
        mapper = DepthMapper(depth_start_m=10.0, depth_end_m=11.0, strip_height_px=100)
        self.assertAlmostEqual(mapper.y_to_depth(50), 10.5)
        self.assertAlmostEqual(mapper.depth_to_y(10.25), 25.0)
        self.assertAlmostEqual(mapper.meters_per_pixel, 0.01)

    def test_missing_ratio(self):
        mapper = DepthMapper(
            depth_start_m=10.0,
            depth_end_m=11.0,
            strip_height_px=100,
            missing_intervals=[MissingInterval(10.20, 10.25)],
        )
        self.assertAlmostEqual(mapper.missing_ratio(10.0, 10.5), 0.1)


if __name__ == "__main__":
    unittest.main()
