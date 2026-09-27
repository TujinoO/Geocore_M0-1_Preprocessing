from __future__ import annotations

import gc
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

from geocore_m1_1.config import M11Config
from geocore_m1_1.io_envi import read_envi_image
from geocore_m1_1.segment_boxes import detect_core_boxes


class OrangeBoxTests(unittest.TestCase):
    def test_detects_complete_non_overlapping_orange_boxes(self) -> None:
        image = np.zeros((1840, 512, 3), dtype=np.uint8)
        for top in (100, 930):
            for y0 in (top, top + 750):
                image[y0:y0 + 40, 4:502] = (115, 65, 20)
                image[y0:y0 + 40, 75:420] = (230, 90, 12)
            image[top:top + 790, 4:20] = (180, 75, 15)
            image[top:top + 790, 485:502] = (180, 75, 15)
        result = detect_core_boxes(image, M11Config(thumbnail_width=512))
        self.assertEqual(result.method, "orange_frame_pairs")
        self.assertEqual(len(result.candidates), 2)
        first, second = [box.bbox_xyxy_raw for box in result.candidates]
        self.assertLess(first[3], second[1])
        self.assertLessEqual(first[0], 4)
        self.assertGreaterEqual(first[2], 501)

    def test_envi_zero_based_default_bands_are_rgb(self) -> None:
        root = Path(tempfile.gettempdir())
        stem = "geocore_rgb_order_" + uuid.uuid4().hex
        hdr = root / f"{stem}.hdr"
        dat = root / f"{stem}.dat"
        try:
            hdr.write_text(
                "ENVI\nsamples = 2\nlines = 1\nbands = 3\n"
                "data type = 1\ninterleave = bil\nbyte order = 0\n"
                "default bands = {2, 1, 0}\n",
                encoding="ascii",
            )
            # BIL: stored channels B, G, R.
            np.array([[[3, 30], [2, 20], [1, 10]]], dtype=np.uint8).tofile(dat)
            image, meta = read_envi_image(dat, hdr)
            pixel = image[0, 0].tolist()
            del image
            gc.collect()
            self.assertEqual(pixel, [1, 2, 3])
            self.assertEqual(meta["rgb_bands_zero_based"], [2, 1, 0])
        finally:
            if dat.exists():
                dat.unlink()
            if hdr.exists():
                hdr.unlink()


if __name__ == "__main__":
    unittest.main()
