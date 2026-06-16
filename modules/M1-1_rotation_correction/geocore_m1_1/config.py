from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class M11Config:
    """Runtime configuration for the M1-1 correction pipeline."""

    thumbnail_width: int = 512
    horizontal_band_left_ratio: float = 0.18
    horizontal_band_right_ratio: float = 0.84
    x_crop_left_ratio: float = 0.17
    x_crop_right_ratio: float = 0.86
    min_box_height_to_width: float = 1.10
    max_box_height_to_width: float = 2.20
    expected_box_height_to_width: float = 1.62
    inter_box_gap_to_width: float = 0.45
    min_gap_to_refine_start: float = 0.10
    boundary_pixel_percentile: float = 88.0
    boundary_profile_percentile: float = 90.0
    local_peak_radius_px: int = 4
    local_peak_min_distance_px: int = 40
    edge_search_margin_px: int = 34
    angle_scan_enabled: bool = True
    angle_scan_range_deg: float = 3.0
    angle_scan_coarse_step_deg: float = 0.05
    angle_scan_fine_step_deg: float = 0.01
    angle_scan_edge_percentile: float = 96.5
    angle_scan_max_points: int = 120_000
    max_abs_angle_deg: float = 10.0
    manual_angle_delta_deg: float = 0.0
    confidence_threshold: float = 0.45
    crop_margin_px: int = 18
    raw_bbox_margin_px: int = 32
    fill_value: int = 0
    save_previews: bool = True
    expected_box_count: int | None = None
    preview_quality: int = 92
