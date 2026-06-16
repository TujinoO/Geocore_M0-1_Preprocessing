from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoreSegment:
    segment_id: str
    core_box_id: str
    hole_id: str
    depth_start_m: float
    depth_end_m: float
    depth_center_m: float
    length_cm: float
    image_path: str
    mask_path: str
    source_strip_bbox: list[int]
    core_coverage_ratio: float
    missing_ratio: float
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "core_box_id": self.core_box_id,
            "hole_id": self.hole_id,
            "depth_start_m": self.depth_start_m,
            "depth_end_m": self.depth_end_m,
            "depth_center_m": self.depth_center_m,
            "length_cm": self.length_cm,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "source_strip_bbox": self.source_strip_bbox,
            "core_coverage_ratio": self.core_coverage_ratio,
            "missing_ratio": self.missing_ratio,
            "quality_flags": self.quality_flags,
        }
