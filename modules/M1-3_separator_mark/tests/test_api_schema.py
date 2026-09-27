import unittest

from geocore_m1_3.api.schemas import SegmentDepthRequest


class ApiSchemaTest(unittest.TestCase):
    def test_default_lane_count_is_automatic(self):
        request = SegmentDepthRequest(
            core_box_id="box_0001",
            m1_2_output_dir="mask",
            output_dir="output",
            hole_id="hole",
            depth_start_m=0,
            depth_end_m=1,
        )
        self.assertEqual(request.layout.lane_count, 0)


if __name__ == "__main__":
    unittest.main()
