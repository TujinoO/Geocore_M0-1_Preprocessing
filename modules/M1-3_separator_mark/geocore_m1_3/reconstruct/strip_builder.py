from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geocore_m1_3.config import ReconstructionConfig
from geocore_m1_3.layout.lane_detector import Lane
from geocore_m1_3.reconstruct.lane_strip import LaneStrip, build_lane_strip


@dataclass
class StripBuildResult:
    strip_rgba: np.ndarray
    strip_mask: np.ndarray
    lane_strips: list[LaneStrip]
    mapping: dict
    warnings: list[dict]


def _standard_width(lanes: list[Lane], mode: str) -> int:
    widths = [max(1, lane.bbox[2] - lane.bbox[0]) for lane in lanes]
    if not widths:
        return 1
    if mode == "median_lane_width":
        return int(np.median(widths))
    return int(max(widths))


def build_reconstructed_strip(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    lanes: list[Lane],
    config: ReconstructionConfig,
) -> StripBuildResult:
    ordered_lanes = sorted(lanes, key=lambda lane: lane.order_index)
    standard_width = _standard_width(ordered_lanes, config.strip_width_mode)
    lane_strips = [
        build_lane_strip(image_rgb=image_rgb, mask=mask, lane=lane, standard_width=standard_width, config=config)
        for lane in ordered_lanes
    ]
    if lane_strips:
        strip_rgba = np.concatenate([lane_strip.image_rgba for lane_strip in lane_strips], axis=0)
        strip_mask = np.concatenate([lane_strip.mask for lane_strip in lane_strips], axis=0)
    else:
        strip_rgba = np.zeros((1, standard_width, 4), dtype=np.uint8)
        strip_mask = np.zeros((1, standard_width), dtype=bool)

    mapping_lanes: list[dict] = []
    y_cursor = 0
    warnings: list[dict] = []
    for lane_strip in lane_strips:
        height = int(lane_strip.image_rgba.shape[0])
        warnings.extend(lane_strip.warnings)
        mapping_lanes.append(
            {
                "lane_id": lane_strip.lane.lane_id,
                "lane_index": lane_strip.lane.lane_index,
                "order_index": lane_strip.lane.order_index,
                "direction": lane_strip.lane.direction,
                "source_bbox": lane_strip.source_bbox,
                "strip_y_start": y_cursor,
                "strip_y_end": y_cursor + height,
                "kept_row_intervals": lane_strip.kept_row_intervals,
            }
        )
        y_cursor += height

    mapping = {
        "strip_width": int(strip_rgba.shape[1]),
        "strip_height": int(strip_rgba.shape[0]),
        "lane_count": len(lane_strips),
        "lanes": mapping_lanes,
    }
    return StripBuildResult(
        strip_rgba=strip_rgba,
        strip_mask=strip_mask,
        lane_strips=lane_strips,
        mapping=mapping,
        warnings=warnings,
    )
