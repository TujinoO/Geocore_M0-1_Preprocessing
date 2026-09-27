from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LayoutOptions(BaseModel):
    lane_count: int = Field(default=0, ge=0, description="0 = infer lane count from the foreground mask")
    lane_order: str = "left_to_right"
    lane_direction: str = "top_to_bottom"


class SegmentationOptions(BaseModel):
    segment_length_cm: float = Field(default=10.0, gt=0)
    overlap_cm: float = Field(default=0.0, ge=0)


class SegmentDepthRequest(BaseModel):
    core_box_id: str
    image_path: str | None = None
    m1_2_output_dir: str
    mask_path: str | None = None
    output_dir: str
    hole_id: str
    depth_start_m: float
    depth_end_m: float
    layout: LayoutOptions = Field(default_factory=LayoutOptions)
    segmentation: SegmentationOptions = Field(default_factory=SegmentationOptions)
    mask_refine: dict[str, Any] = Field(default_factory=dict)
    reconstruction: dict[str, Any] = Field(default_factory=dict)
    depth_mapping: dict[str, Any] = Field(default_factory=dict)
    missing_intervals: list[dict[str, Any]] = Field(default_factory=list)
    depth_anchors: list[dict[str, Any]] = Field(default_factory=list)
    gap_policy: Literal["close_artificial_gaps", "preserve_all_gaps"] | None = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    status_url: str | None = None
    output_dir: str | None = None
    output_files: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


class DepthMetadataRequest(BaseModel):
    project_id: str | None = None
    hole_id: str
    core_box_id: str
    box_no: str | None = None
    run_no: str | None = None
    depth_start_m: float
    depth_end_m: float
    missing_intervals: list[dict[str, Any]] = Field(default_factory=list)
    depth_anchors: list[dict[str, Any]] = Field(default_factory=list)
    operator: str | None = None
    notes: str | None = None
