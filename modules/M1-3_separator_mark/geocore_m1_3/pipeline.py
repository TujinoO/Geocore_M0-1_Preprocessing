from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from geocore_m1_3 import __version__
from geocore_m1_3.config import PipelineConfig, config_from_dict, deep_update, load_config
from geocore_m1_3.depth.mapper import DepthMapper, parse_depth_anchors, parse_missing_intervals
from geocore_m1_3.io.m1_2_loader import load_m1_2_inputs
from geocore_m1_3.layout.lane_detector import detect_lanes
from geocore_m1_3.mask_refine.refiner import refine_mask
from geocore_m1_3.quality.reports import build_quality_report
from geocore_m1_3.reconstruct.strip_builder import build_reconstructed_strip
from geocore_m1_3.segments.cutter import cut_core_segments
from geocore_m1_3.utils.image_io import (
    make_overlay,
    read_mask,
    read_rgb,
    save_lane_debug,
    save_mask,
    save_rgb,
    save_rgba,
)
from geocore_m1_3.utils.json_io import write_json


def _new_task_id() -> str:
    return "m1_3_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _config_from_request(payload: dict[str, Any]) -> PipelineConfig:
    config_path = payload.get("config_path")
    config = load_config(config_path) if config_path else config_from_dict()
    merged = config.to_dict()
    for section in ["layout", "mask_refine", "reconstruction", "depth_mapping", "segmentation"]:
        if isinstance(payload.get(section), dict):
            deep_update(merged[section], payload[section])
    if payload.get("gap_policy"):
        merged["reconstruction"]["gap_policy"] = payload["gap_policy"]
    return config_from_dict(merged)


def _resolve_image_path(payload: dict[str, Any], metadata: dict) -> Path:
    image_path = payload.get("image_path") or metadata.get("input_path")
    if not image_path:
        raise ValueError("image_path is required when M1-2 metadata does not contain input_path.")
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {path}")
    return path


def _validate_dimensions(image_shape: tuple[int, int, int], mask_shape: tuple[int, int]) -> None:
    image_height, image_width = image_shape[:2]
    mask_height, mask_width = mask_shape
    if image_width != mask_width or image_height != mask_height:
        raise ValueError(
            "Image and mask dimensions do not match: "
            f"image={image_width}x{image_height}, mask={mask_width}x{mask_height}"
        )


def run_segment_depth(payload: dict[str, Any]) -> dict:
    """Run the complete M1-3 segmentation and depth marking pipeline."""
    if "m1_2_output_dir" not in payload and "mask_path" not in payload:
        raise ValueError("m1_2_output_dir or mask_path is required.")
    if "output_dir" not in payload:
        raise ValueError("output_dir is required.")
    if "depth_start_m" not in payload or "depth_end_m" not in payload:
        raise ValueError("depth_start_m and depth_end_m are required.")

    config = _config_from_request(payload)
    task_id = payload.get("task_id") or _new_task_id()
    output_root = Path(payload["output_dir"]) / task_id
    output_root.mkdir(parents=True, exist_ok=True)

    m1_2_dir = payload.get("m1_2_output_dir") or Path(payload["mask_path"]).parent
    m1_2_inputs = load_m1_2_inputs(m1_2_dir, payload.get("mask_path"))
    image_path = _resolve_image_path(payload, m1_2_inputs.metadata)
    image_rgb = read_rgb(image_path)
    raw_mask = read_mask(m1_2_inputs.mask_path)
    _validate_dimensions(image_rgb.shape, raw_mask.shape)

    hole_id = str(payload.get("hole_id", "UNKNOWN_HOLE"))
    core_box_id = str(payload.get("core_box_id", Path(image_path).stem))
    depth_start_m = float(payload["depth_start_m"])
    depth_end_m = float(payload["depth_end_m"])
    if depth_end_m <= depth_start_m:
        raise ValueError("depth_end_m must be greater than depth_start_m.")

    input_manifest = {
        "task_id": task_id,
        "module": "M1-3 core segmentation and depth marking",
        "module_version": __version__,
        "image_path": str(image_path),
        "m1_2_output_dir": str(m1_2_dir),
        "mask_path": str(m1_2_inputs.mask_path),
        "metadata_path": str(m1_2_inputs.metadata_path) if m1_2_inputs.metadata_path else None,
        "contours_path": str(m1_2_inputs.contours_path) if m1_2_inputs.contours_path else None,
        "probability_path": str(m1_2_inputs.probability_path) if m1_2_inputs.probability_path else None,
        "hole_id": hole_id,
        "core_box_id": core_box_id,
        "depth_start_m": depth_start_m,
        "depth_end_m": depth_end_m,
        "config": config.to_dict(),
        "request_payload": payload,
    }
    write_json(output_root / "input_manifest.json", input_manifest)

    initial_lanes = detect_lanes(raw_mask, config.layout, core_box_id=core_box_id)
    refine_result = refine_mask(raw_mask, initial_lanes, config.mask_refine)
    refined_mask = refine_result.refined_mask
    lanes = detect_lanes(refined_mask, config.layout, core_box_id=core_box_id)

    save_mask(output_root / "refined_mask.png", refined_mask)
    save_mask(output_root / "refined_mask.tif", refined_mask)
    save_rgb(output_root / "refined_overlay.png", make_overlay(image_rgb, refined_mask))
    save_lane_debug(output_root / "lane_debug_overlay.png", image_rgb, [lane.to_dict() for lane in lanes])
    write_json(output_root / "lane_detection.json", {"lanes": [lane.to_dict() for lane in lanes]})
    write_json(
        output_root / "removed_components.json",
        {
            "removed_component_count": len(refine_result.removed_components),
            "components": [component.to_dict(include_runs=False) for component in refine_result.removed_components],
        },
    )
    write_json(output_root / "mask_refine_report.json", refine_result.report)

    strip_result = build_reconstructed_strip(image_rgb, refined_mask, lanes, config.reconstruction)
    lane_strip_dir = output_root / "lane_strips"
    lane_strip_dir.mkdir(parents=True, exist_ok=True)
    for lane_strip in strip_result.lane_strips:
        idx = lane_strip.lane.lane_index
        save_rgba(lane_strip_dir / f"lane_{idx:02d}.png", lane_strip.image_rgba)
        save_mask(lane_strip_dir / f"lane_{idx:02d}_mask.png", lane_strip.mask)
    save_rgba(output_root / "reconstructed_strip.png", strip_result.strip_rgba)
    save_mask(output_root / "reconstructed_strip_mask.png", strip_result.strip_mask)
    # JPEG preview cannot carry alpha; black background is acceptable for fast visual review.
    preview_rgb = strip_result.strip_rgba[:, :, :3].copy()
    save_rgb(output_root / "reconstructed_strip_preview.jpg", preview_rgb)
    write_json(output_root / "strip_mapping.json", strip_result.mapping)

    depth_mapper = DepthMapper(
        depth_start_m=depth_start_m,
        depth_end_m=depth_end_m,
        strip_height_px=int(strip_result.strip_mask.shape[0]),
        anchors=parse_depth_anchors(payload.get("depth_anchors")),
        missing_intervals=parse_missing_intervals(payload.get("missing_intervals")),
    )
    depth_mapping = depth_mapper.to_dict()
    write_json(output_root / "depth_mapping.json", depth_mapping)

    segments = cut_core_segments(
        strip_rgba=strip_result.strip_rgba,
        strip_mask=strip_result.strip_mask,
        mapper=depth_mapper,
        output_dir=output_root,
        hole_id=hole_id,
        core_box_id=core_box_id,
        config=config.segmentation,
    )
    segments_json = {
        "task_id": task_id,
        "module": "M1-3 core segmentation and depth marking",
        "module_version": __version__,
        "core_box_id": core_box_id,
        "hole_id": hole_id,
        "depth_start_m": depth_start_m,
        "depth_end_m": depth_end_m,
        "segment_length_cm": config.segmentation.segment_length_cm,
        "overlap_cm": config.segmentation.overlap_cm,
        "segment_count": len(segments),
        "segments": [segment.to_dict() for segment in segments],
    }
    write_json(output_root / "segments.json", segments_json)

    all_warnings = refine_result.warnings + strip_result.warnings
    quality_report = build_quality_report(
        lane_count_expected=config.layout.lane_count,
        lane_count_detected=len(lanes),
        refine_report=refine_result.report,
        strip_mapping=strip_result.mapping,
        depth_mapping=depth_mapping,
        segments=segments,
        warnings=all_warnings,
    )
    write_json(output_root / "quality_report.json", quality_report)
    write_json(
        output_root / "review_manifest.json",
        {
            "task_id": task_id,
            "needs_review": bool(all_warnings),
            "warnings": all_warnings,
            "review_targets": {
                "lane_debug_overlay": str(output_root / "lane_debug_overlay.png"),
                "refined_overlay": str(output_root / "refined_overlay.png"),
                "reconstructed_strip_preview": str(output_root / "reconstructed_strip_preview.jpg"),
            },
        },
    )

    return {
        "task_id": task_id,
        "status": "succeeded",
        "output_dir": str(output_root),
        "output_files": {
            "refined_mask": str(output_root / "refined_mask.png"),
            "lane_detection": str(output_root / "lane_detection.json"),
            "reconstructed_strip": str(output_root / "reconstructed_strip.png"),
            "segments_json": str(output_root / "segments.json"),
            "quality_report": str(output_root / "quality_report.json"),
        },
        "metrics": {
            "lane_count_detected": len(lanes),
            "segment_count": len(segments),
            "removed_component_count": refine_result.report.get("removed_component_count", 0),
            "strip_height_px": int(strip_result.strip_mask.shape[0]),
            "meters_per_pixel": depth_mapping["meters_per_pixel"],
        },
        "warnings": all_warnings,
    }
