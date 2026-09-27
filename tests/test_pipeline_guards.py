from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

from geocore_preprocessing.fusion_accessor import ImageGrid
from geocore_preprocessing.pipeline import _box_depth_windows, _resolve_m11_input


class PipelineGuardsTest(unittest.TestCase):
    def test_downsampled_preview_cannot_replace_native_rgb(self):
        accessor = Mock()
        accessor.root = Path(__file__).parent
        accessor.manifest = {"metadata": {}}
        accessor.grid = ImageGrid(width=100, height=200, name="rgb")
        accessor.preview_path.return_value = Path(__file__)
        accessor.preview_grid.return_value = ImageGrid(width=10, height=20, name="preview")
        with self.assertRaisesRegex(ValueError, "downsampled"):
            _resolve_m11_input(accessor, {})

    def test_per_box_depths_must_match_detected_order(self):
        boxes = [{"box_id": "box_0001"}, {"box_id": "box_0002"}]
        self.assertEqual(
            _box_depth_windows({"box_depths": [[10, 11], [11, 12]]}, boxes, 10, 12, []),
            [(10.0, 11.0), (11.0, 12.0)],
        )
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            _box_depth_windows({"box_depths": [[10, 11.5], [11, 12]]}, boxes, 10, 12, [])
