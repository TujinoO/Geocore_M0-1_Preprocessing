from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CoreBoxCandidate:
    box_id: str
    order_index: int
    bbox_xyxy_thumb: tuple[int, int, int, int]
    bbox_xyxy_raw: tuple[int, int, int, int]
    boundary_score: float
    split_score: float


@dataclass(slots=True)
class AngleEstimate:
    angle_deg: float
    confidence: float
    left_slope: float | None
    right_slope: float | None
    support_rows: int
    method: str


@dataclass(slots=True)
class CoreBoxResult:
    box_id: str
    order_index: int
    bbox_xyxy_raw: tuple[int, int, int, int]
    bbox_xyxy_corrected: tuple[int, int, int, int]
    angle_deg: float
    confidence: float
    needs_manual_review: bool
    rotation_matrix_2x3: list[list[float]]
    output_image: str
    output_mask: str
    preview_image: str | None = None
    review_source_image: str | None = None
    source_crop_bbox_raw: tuple[int, int, int, int] | None = None
    review_source_size_px: tuple[int, int] | None = None
    review_source_scale_xy: tuple[float, float] = (1.0, 1.0)
    correction_status: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class M11BatchResult:
    module: str
    input_path: str
    hdr_path: str | None
    output_dir: str
    box_count: int
    boxes: list[CoreBoxResult]
    detection_method: str = "horizontal_gradient_peaks"
    detection_preview: str | None = None
    qa_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["boxes"] = [box.to_dict() for box in self.boxes]
        return data
