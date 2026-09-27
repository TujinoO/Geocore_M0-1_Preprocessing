from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geocore_m1_3.config import ReconstructionConfig
from geocore_m1_3.layout.lane_detector import Lane


@dataclass
class LaneStrip:
    lane: Lane
    image_rgba: np.ndarray
    mask: np.ndarray
    source_bbox: list[int]
    kept_row_intervals: list[list[int]]
    warnings: list[dict]


def _active_row_intervals(active: np.ndarray, padding: int, height: int) -> list[list[int]]:
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return [[0, height]]
    intervals: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        intervals.append([max(0, start - padding), min(height, prev + 1 + padding)])
        start = prev = value
    intervals.append([max(0, start - padding), min(height, prev + 1 + padding)])

    merged: list[list[int]] = []
    for interval in intervals:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def build_lane_strip(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    lane: Lane,
    standard_width: int,
    config: ReconstructionConfig,
) -> LaneStrip:
    x0, y0, x1, y1 = lane.bbox
    crop_rgb = image_rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if lane.direction == "bottom_to_top":
        crop_rgb = crop_rgb[::-1]
        crop_mask = crop_mask[::-1]

    warnings: list[dict] = []
    local_height, local_width = crop_mask.shape
    if config.gap_policy not in {"close_artificial_gaps", "preserve_all_gaps"}:
        raise ValueError(f"Unsupported gap_policy: {config.gap_policy}")
    row_counts = crop_mask.sum(axis=1)
    active = row_counts > max(1, int(local_width * config.row_keep_min_coverage))
    intervals = _active_row_intervals(active, config.row_keep_padding_px, local_height)
    for before, after in zip(intervals, intervals[1:]):
        gap = after[0] - before[1]
        if gap >= config.large_gap_warning_px:
            compressed = config.gap_policy == "close_artificial_gaps"
            warnings.append({
                "code": "large_gap_requires_review" if compressed else "large_gap_preserved_requires_review",
                "message": (
                    "A large blank interval was compressed inside a lane; confirm whether it is lost core."
                    if compressed else
                    "A large blank interval was preserved inside a lane; confirm whether it represents lost core."
                ),
                "lane_index": lane.lane_index,
                "gap_pixels": int(gap),
            })
    if config.gap_policy == "close_artificial_gaps":
        keep_indices = np.concatenate([np.arange(start, end) for start, end in intervals if end > start])
        crop_rgb = crop_rgb[keep_indices]
        crop_mask = crop_mask[keep_indices]
        kept_row_intervals = intervals
    else:
        kept_row_intervals = [[0, local_height]]

    if crop_rgb.size == 0:
        crop_rgb = np.zeros((1, local_width, 3), dtype=np.uint8)
        crop_mask = np.zeros((1, local_width), dtype=bool)

    alpha = crop_mask.astype(np.uint8) * 255
    rgba = np.dstack([crop_rgb, alpha])
    canvas_width = max(1, int(standard_width))
    canvas = np.zeros((rgba.shape[0], canvas_width, 4), dtype=np.uint8)
    paste_x = max(0, (canvas_width - rgba.shape[1]) // 2)
    paste_x1 = min(canvas_width, paste_x + rgba.shape[1])
    src_x1 = paste_x1 - paste_x
    canvas[:, paste_x:paste_x1] = rgba[:, :src_x1]

    mask_canvas = np.zeros((crop_mask.shape[0], canvas_width), dtype=bool)
    mask_canvas[:, paste_x:paste_x1] = crop_mask[:, :src_x1]
    return LaneStrip(
        lane=lane,
        image_rgba=canvas,
        mask=mask_canvas,
        source_bbox=lane.bbox,
        kept_row_intervals=kept_row_intervals,
        warnings=warnings,
    )
