from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import M11Config
from .correction import correct_core_box
from .geometry import estimate_box_angle
from .io_envi import read_input_image, save_rgb_image
from .pipeline import _crop_for_angle


def load_metadata(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found under {output_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(output_dir) / "metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def get_review_items(output_dir: str | Path, only_needs_review: bool = True) -> list[dict[str, Any]]:
    metadata = load_metadata(output_dir)
    boxes = metadata.get("boxes", [])
    if only_needs_review:
        boxes = [box for box in boxes if box.get("needs_manual_review")]
    return boxes


def manual_correct_box(
    output_dir: str | Path,
    input_path: str,
    hdr_path: str | None,
    box_id: str,
    annotation: dict[str, Any],
    manual_angle_delta_deg: float = 0.0,
) -> dict[str, Any]:
    """Apply a user rectangle/polygon annotation and synchronize corrected output."""

    output_root = Path(output_dir)
    metadata = load_metadata(output_root)
    boxes = metadata.get("boxes", [])
    target = next((box for box in boxes if box.get("box_id") == box_id), None)
    if target is None:
        raise ValueError(f"Unknown box_id: {box_id}")
    source_bbox = target.get("source_crop_bbox_raw")
    if not source_bbox:
        raise ValueError(f"{box_id} has no source_crop_bbox_raw; rerun auto pipeline first.")

    points = _annotation_points(annotation)
    if len(points) < 2:
        raise ValueError("Manual annotation requires at least two points.")
    crop_x0, crop_y0, crop_x1, crop_y1 = [int(v) for v in source_bbox]
    source_w = crop_x1 - crop_x0 + 1
    source_h = crop_y1 - crop_y0 + 1
    clipped = [(max(0.0, min(source_w - 1.0, x)), max(0.0, min(source_h - 1.0, y))) for x, y in points]
    xs = [point[0] for point in clipped]
    ys = [point[1] for point in clipped]
    raw_bbox = (
        crop_x0 + int(math.floor(min(xs))),
        crop_y0 + int(math.floor(min(ys))),
        crop_x0 + int(math.ceil(max(xs))),
        crop_y0 + int(math.ceil(max(ys))),
    )

    image, _ = read_input_image(input_path, hdr_path)
    cfg = M11Config(manual_angle_delta_deg=manual_angle_delta_deg)
    polygon_angle = _angle_from_polygon(clipped) if len(clipped) >= 4 else None
    if polygon_angle is None:
        crop, inner = _crop_for_angle(image, raw_bbox, cfg)
        angle = estimate_box_angle(crop, inner, cfg)
        angle_deg = angle.angle_deg
        confidence = angle.confidence
    else:
        angle_deg = max(-cfg.max_abs_angle_deg, min(cfg.max_abs_angle_deg, polygon_angle + manual_angle_delta_deg))
        confidence = 1.0

    corrected_image, corrected_mask, corrected_bbox, matrix, _ = correct_core_box(
        image,
        raw_bbox,
        angle_deg,
        cfg,
    )
    corrected_dir = output_root / "corrected_boxes"
    masks_dir = output_root / "masks"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image_path = corrected_dir / f"{box_id}.png"
    mask_path = masks_dir / f"{box_id}_mask.png"
    save_rgb_image(image_path, corrected_image, quality=cfg.preview_quality)
    save_rgb_image(mask_path, corrected_mask, quality=cfg.preview_quality)

    target.update(
        {
            "bbox_xyxy_raw": list(raw_bbox),
            "bbox_xyxy_corrected": list(corrected_bbox),
            "angle_deg": float(angle_deg),
            "confidence": float(confidence),
            "needs_manual_review": False,
            "rotation_matrix_2x3": matrix,
            "output_image": str(image_path),
            "output_mask": str(mask_path),
            "correction_status": "manual",
            "manual_annotation": {
                "type": annotation.get("type", "polygon" if len(clipped) > 2 else "rectangle"),
                "points_source_px": [[float(x), float(y)] for x, y in clipped],
                "source_crop_bbox_raw": source_bbox,
            },
        }
    )
    save_metadata(output_root, metadata)
    write_qa_report_from_metadata(output_root, metadata)
    return target


def write_qa_report_from_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> str:
    output_root = Path(output_dir)
    path = output_root / "qa_report.md"
    lines = [
        "# M1-1 QA Report",
        "",
        f"- input: `{metadata.get('input_path')}`",
        f"- hdr: `{metadata.get('hdr_path')}`",
        f"- box_count: `{metadata.get('box_count')}`",
        "",
        "| box_id | angle_deg | confidence | review | status | output |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for box in metadata.get("boxes", []):
        review = "yes" if box.get("needs_manual_review") else "no"
        status = box.get("correction_status", "auto")
        lines.append(
            f"| {box.get('box_id')} | {float(box.get('angle_deg', 0.0)):.4f} | "
            f"{float(box.get('confidence', 0.0)):.3f} | {review} | {status} | `{box.get('output_image')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _annotation_points(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    if "points" in annotation:
        return [(float(item["x"]), float(item["y"])) for item in annotation["points"]]
    if "rect" in annotation:
        rect = annotation["rect"]
    else:
        rect = annotation
    if {"x", "y", "width", "height"}.issubset(rect):
        x = float(rect["x"])
        y = float(rect["y"])
        w = float(rect["width"])
        h = float(rect["height"])
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if {"x0", "y0", "x1", "y1"}.issubset(rect):
        x0 = float(rect["x0"])
        y0 = float(rect["y0"])
        x1 = float(rect["x1"])
        y1 = float(rect["y1"])
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    raise ValueError("Annotation must contain points, rect{x,y,width,height}, or x0/y0/x1/y1.")


def _angle_from_polygon(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 4:
        return None
    pts = np.asarray(points, dtype=np.float64)
    median_x = float(np.median(pts[:, 0]))
    left = pts[pts[:, 0] <= median_x]
    right = pts[pts[:, 0] > median_x]
    slopes: list[float] = []
    for side in (left, right):
        if len(side) < 2 or float(np.ptp(side[:, 1])) < 1.0:
            continue
        slope, _ = np.polyfit(side[:, 1], side[:, 0], deg=1)
        slopes.append(float(slope))
    if not slopes:
        return None
    return -math.degrees(math.atan(float(np.mean(slopes))))
