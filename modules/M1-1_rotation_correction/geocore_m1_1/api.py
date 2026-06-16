from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import M11Config
from .models import M11BatchResult
from .pipeline import run_m11_rotation_correction
from .review import get_review_items, load_metadata, manual_correct_box


@dataclass(slots=True)
class RotationCorrectionRequest:
    input_path: str
    hdr_path: str | None = None
    output_dir: str = "outputs/m1_1"
    manual_angle_delta_deg: float = 0.0
    save_preview: bool = True
    expected_box_count: int | None = None


def correct_from_request(request: RotationCorrectionRequest) -> M11BatchResult:
    """Small framework-neutral adapter for backend integration."""

    config = M11Config(
        manual_angle_delta_deg=request.manual_angle_delta_deg,
        save_previews=request.save_preview,
        expected_box_count=request.expected_box_count,
    )
    return run_m11_rotation_correction(
        input_path=request.input_path,
        hdr_path=request.hdr_path,
        output_dir=request.output_dir,
        config=config,
    )


def run_auto_correction(payload: dict[str, Any]) -> dict[str, Any]:
    request = RotationCorrectionRequest(
        input_path=str(payload["input_path"]),
        hdr_path=payload.get("hdr_path"),
        output_dir=str(payload.get("output_dir", "outputs/m1_1")),
        manual_angle_delta_deg=float(payload.get("manual_angle_delta_deg", 0.0)),
        save_preview=bool(payload.get("save_preview", True)),
        expected_box_count=payload.get("expected_box_count"),
    )
    result = correct_from_request(request)
    return result.to_dict()


def read_result_metadata(output_dir: str | Path) -> dict[str, Any]:
    return load_metadata(output_dir)


def read_review_items(output_dir: str | Path, only_needs_review: bool = True) -> dict[str, Any]:
    items = get_review_items(output_dir, only_needs_review=only_needs_review)
    return {"output_dir": str(output_dir), "count": len(items), "items": items}


def submit_manual_correction(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("output_dir", "input_path", "box_id", "annotation")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    result = manual_correct_box(
        output_dir=str(payload["output_dir"]),
        input_path=str(payload["input_path"]),
        hdr_path=payload.get("hdr_path"),
        box_id=str(payload["box_id"]),
        annotation=dict(payload["annotation"]),
        manual_angle_delta_deg=float(payload.get("manual_angle_delta_deg", 0.0)),
    )
    return {"status": "success", "box": result}
