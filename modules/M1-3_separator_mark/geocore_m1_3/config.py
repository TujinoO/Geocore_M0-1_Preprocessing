from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass
class LayoutConfig:
    lane_count: int = 5
    lane_order: str = "left_to_right"
    lane_direction: str = "top_to_bottom"
    allow_snake_order: bool = False
    lane_padding_px: int = 8


@dataclass
class MaskRefineConfig:
    enable: bool = True
    min_component_area_px: int = 200
    fill_hole_area_px: int = 800
    smooth_kernel_size: int = 3
    outside_lane_remove: bool = True
    lane_margin_px: int = 20
    removed_area_warning_ratio: float = 0.05


@dataclass
class ReconstructionConfig:
    strip_width_mode: str = "max_lane_width"
    background: str = "transparent"
    gap_policy: str = "close_artificial_gaps"
    row_keep_min_coverage: float = 0.015
    row_keep_padding_px: int = 2
    large_gap_warning_px: int = 80


@dataclass
class DepthMappingConfig:
    mode: str = "linear"
    require_depth_metadata: bool = True
    max_depth_error_cm: float = 1.0


@dataclass
class SegmentationConfig:
    segment_length_cm: float = 10.0
    overlap_cm: float = 0.0
    min_core_coverage_ratio: float = 0.2


@dataclass
class PipelineConfig:
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    mask_refine: MaskRefineConfig = field(default_factory=MaskRefineConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    depth_mapping: DepthMappingConfig = field(default_factory=DepthMappingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def config_from_dict(data: Mapping[str, Any] | None = None) -> PipelineConfig:
    merged = PipelineConfig().to_dict()
    if data:
        deep_update(merged, data)
    return PipelineConfig(
        layout=LayoutConfig(**merged["layout"]),
        mask_refine=MaskRefineConfig(**merged["mask_refine"]),
        reconstruction=ReconstructionConfig(**merged["reconstruction"]),
        depth_mapping=DepthMappingConfig(**merged["depth_mapping"]),
        segmentation=SegmentationConfig(**merged["segmentation"]),
    )


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> PipelineConfig:
    data: dict[str, Any] = {}
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    if overrides:
        deep_update(data, overrides)
    return config_from_dict(data)
