from __future__ import annotations

import unittest
from pathlib import Path

from geocore_m1_1.io_envi import parse_envi_header, read_envi_image


ROOT = Path(__file__).resolve().parents[1]


class EnviReaderTests(unittest.TestCase):
    def test_parse_sample_header(self) -> None:
        meta = parse_envi_header(ROOT / "RGB-20230909_141858-00000.hdr")
        self.assertEqual(meta["samples"], 2048)
        self.assertEqual(meta["lines"], 22480)
        self.assertEqual(meta["bands"], 3)
        self.assertEqual(meta["interleave"], "bil")

    def test_read_sample_memmap_shape(self) -> None:
        image, meta = read_envi_image(
            ROOT / "RGB-20230909_141858-00000.dat",
            ROOT / "RGB-20230909_141858-00000.hdr",
        )
        self.assertEqual(image.shape, (22480, 2048, 3))
        self.assertEqual(meta["shape_hwc"], (22480, 2048, 3))


if __name__ == "__main__":
    unittest.main()

