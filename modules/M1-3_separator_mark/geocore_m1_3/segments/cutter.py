from __future__ import annotations

from pathlib import Path

import numpy as np

from geocore_m1_3.config import SegmentationConfig
from geocore_m1_3.depth.mapper import DepthMapper
from geocore_m1_3.segments.core_segment import CoreSegment
from geocore_m1_3.segments.naming import segment_id
from geocore_m1_3.utils.image_io import save_mask, save_rgba


def _depth_windows(depth_start_m: float, depth_end_m: float, segment_length_m: float, overlap_m: float) -> list[tuple[float, float]]:
    if segment_length_m <= 0:
        raise ValueError("segment_length_cm must be positive.")
    if overlap_m < 0:
        raise ValueError("overlap_cm cannot be negative.")
    if overlap_m >= segment_length_m:
        raise ValueError("overlap_cm must be smaller than segment_length_cm.")
    step = segment_length_m - overlap_m
    windows: list[tuple[float, float]] = []
    current = depth_start_m
    eps = 1e-9
    while current < depth_end_m - eps:
        end = min(depth_end_m, current + segment_length_m)
        windows.append((round(current, 6), round(end, 6)))
        if end >= depth_end_m - eps:
            break
        current += step
    return windows


def cut_core_segments(
    strip_rgba: np.ndarray,
    strip_mask: np.ndarray,
    mapper: DepthMapper,
    output_dir: str | Path,
    hole_id: str,
    core_box_id: str,
    config: SegmentationConfig,
) -> list[CoreSegment]:
    output_root = Path(output_dir)
    segments_dir = output_root / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    segment_length_m = config.segment_length_cm / 100.0
    overlap_m = config.overlap_cm / 100.0
    windows = _depth_windows(mapper.depth_start_m, mapper.depth_end_m, segment_length_m, overlap_m)

    segments: list[CoreSegment] = []
    height, width = strip_mask.shape
    for depth_start, depth_end in windows:
        y0 = int(np.floor(mapper.depth_to_y(depth_start)))
        y1 = int(np.ceil(mapper.depth_to_y(depth_end)))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        crop_rgba = strip_rgba[y0:y1, :, :]
        crop_mask = strip_mask[y0:y1, :]

        sid = segment_id(hole_id, depth_start, depth_end)
        image_path = segments_dir / f"{sid}.png"
        mask_path = segments_dir / f"{sid}_mask.png"
        save_rgba(image_path, crop_rgba)
        save_mask(mask_path, crop_mask)

        total_pixels = max(1, int(crop_mask.size))
        coverage = float(crop_mask.sum() / total_pixels)
        missing_ratio = mapper.missing_ratio(depth_start, depth_end)
        flags: list[str] = []
        if coverage < config.min_core_coverage_ratio:
            flags.append("low_core_coverage")
        if missing_ratio > 0:
            flags.append("contains_missing_interval")
        if (depth_end - depth_start) * 100.0 < config.segment_length_cm - 1e-6:
            flags.append("partial_length")

        segments.append(
            CoreSegment(
                segment_id=sid,
                core_box_id=core_box_id,
                hole_id=hole_id,
                depth_start_m=float(depth_start),
                depth_end_m=float(depth_end),
                depth_center_m=float((depth_start + depth_end) / 2.0),
                length_cm=float((depth_end - depth_start) * 100.0),
                image_path=str(image_path),
                mask_path=str(mask_path),
                source_strip_bbox=[0, y0, int(width), y1],
                core_coverage_ratio=coverage,
                missing_ratio=missing_ratio,
                quality_flags=flags,
            )
        )
    return segments
