from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from geocore_preprocessing.fusion_accessor import ImageGrid
from geocore_preprocessing.pipeline import _box_depth_windows, _preflight_m0_output, _resolve_m11_input, _run_m13_for_box


class PipelineGuardsTest(unittest.TestCase):
    def test_per_box_reviewed_dividers_reach_m13(self):
        box = {"box_id": "box_0002", "output_image": "corrected.png"}
        payload = {"lane_dividers_by_box": {"box_0002": [100, 200]}}
        with patch("geocore_m1_3.pipeline.run_segment_depth", return_value={"status": "succeeded"}) as run:
            _run_m13_for_box(box, {"output_dir": "mask"}, payload, Path("out"),
                             hole_id="H", core_box_id="H_box_0002",
                             depth_start_m=0.0, depth_end_m=1.0)
        self.assertEqual(run.call_args.args[0]["layout"]["lane_dividers_x"], [100, 200])

    def test_only_eligible_matching_source_prior_reaches_m13(self):
        source = r"H:\RGB\hole01.dat"
        box = {"box_id": "box_0002", "output_image": "corrected.png"}
        payload = {"source_slot_priors": {
            "schema": "geocore_source_slot_priors.v1",
            "sources": {source: {"physical_slot_count": 5, "support_sample_ids": ["sample_014", "sample_015"],
                                 "count_prior_eligible": True}},
        }}
        with patch("geocore_m1_3.pipeline.run_segment_depth", return_value={"status": "succeeded"}) as run:
            _run_m13_for_box(box, {"output_dir": "mask"}, payload, Path("out"),
                             hole_id="H", core_box_id="H_box_0002",
                             depth_start_m=0.0, depth_end_m=1.0, m11_source_path=source)
        self.assertEqual(run.call_args.args[0]["source_slot_prior"]["physical_slot_count"], 5)
        self.assertNotIn("lane_dividers_x", run.call_args.args[0].get("layout", {}))

    def test_m0_preflight_accounts_for_padded_zarr_chunks(self):
        metadata = [
            SimpleNamespace(lines=768, samples=512, bands=3),
            SimpleNamespace(lines=149, samples=81, bands=341),
            SimpleNamespace(lines=149, samples=81, bands=212),
        ]
        payload = {"rgb_hdr": "rgb.hdr", "nir_hdr": "nir.hdr", "swir_hdr": "swir.hdr", "m0_streaming": True}
        with patch("geocore_m01_fusion.envi.parse_envi_header", side_effect=metadata), \
             patch("geocore_preprocessing.pipeline.shutil.disk_usage", return_value=SimpleNamespace(free=10_000_000_000)):
            self.assertEqual(_preflight_m0_output(payload, Path(".")), 1_207_959_552)
        with patch("geocore_m01_fusion.envi.parse_envi_header", side_effect=metadata), \
             patch("geocore_preprocessing.pipeline.shutil.disk_usage", return_value=SimpleNamespace(free=1_000_000_000)):
            with self.assertRaisesRegex(RuntimeError, "cannot be safely materialized"):
                _preflight_m0_output(payload, Path("."))

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
