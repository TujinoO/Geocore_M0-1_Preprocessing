from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from geocore_m1_3 import __version__
from geocore_m1_3.config import PipelineConfig, config_from_dict, deep_update, load_config
from geocore_m1_3.depth.mapper import DepthMapper, parse_depth_anchors, parse_missing_intervals
from geocore_m1_3.io.m1_2_loader import load_m1_2_inputs
from geocore_m1_3.layout.frame_bounds import trim_orange_frame_rows
from geocore_m1_3.layout.lane_detector import detect_lanes, estimate_lane_count
from geocore_m1_3.layout.rgb_slot_geometry import estimate_dark_rail_geometry, estimate_rgb_slot_geometry
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


def _save_rgb_slot_suggestion(image_rgb, dividers_x: list[int], path: Path) -> None:
    """Draw unreviewed RGB dividers at bounded preview resolution."""

    preview = Image.fromarray(image_rgb[:, :, :3], mode="RGB")
    original_width = preview.width
    preview.thumbnail((1600, 2400), Image.Resampling.BOX)
    draw = ImageDraw.Draw(preview)
    for index, x in enumerate(dividers_x, start=1):
        px = int(round(x * preview.width / original_width))
        draw.line((px, 0, px, preview.height), fill=(255, 55, 55), width=3)
        draw.text((min(px + 5, preview.width - 40), 24), str(index), fill=(255, 255, 255))
    draw.rectangle((0, 0, min(preview.width, 290), 21), fill=(25, 25, 25))
    draw.text((5, 4), "UNREVIEWED RGB SUGGESTION", fill=(255, 215, 0))
    preview.save(path, quality=88)


def _physical_slot_warnings(source_slot_prior: dict | None, lane_count: int,
                            reviewed_dividers: bool) -> list[dict]:
    warnings = []
    if source_slot_prior is not None and int(source_slot_prior["physical_slot_count"]) != lane_count:
        warnings.append({
            "code": "source_slot_count_prior_disagreement",
            "message": "The slot count differs from manually labelled boxes of this RGB source; verify this box before using the reconstruction.",
            "source_prior_count": int(source_slot_prior["physical_slot_count"]),
            "current_box_count": lane_count,
            "support_sample_ids": source_slot_prior.get("support_sample_ids", []),
        })
    if not reviewed_dividers:
        warnings.append({
            "code": "physical_slot_count_unverified",
            "message": "Neither RGB candidates nor the foreground mask are a reviewed physical slot label for this box.",
        })
    return warnings


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
    frame_trim = trim_orange_frame_rows(raw_mask, image_rgb)

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
        "orange_frame_trim": frame_trim,
    }
    write_json(output_root / "input_manifest.json", input_manifest)

    raw_lane_estimate = estimate_lane_count(raw_mask)
    source_slot_prior = payload.get("source_slot_prior")
    if source_slot_prior is not None:
        if not isinstance(source_slot_prior, dict) or not 2 <= int(source_slot_prior.get("physical_slot_count", 0)) <= 8:
            raise ValueError("source_slot_prior must contain a physical_slot_count from 2 to 8")
    rgb_slot_candidate = estimate_rgb_slot_geometry(image_rgb, raw_mask)
    dark_rail_candidate = estimate_dark_rail_geometry(image_rgb)
    suggestion_path = None
    if dark_rail_candidate is not None:
        suggestion_path = output_root / "rgb_dark_rail_suggestion.jpg"
        _save_rgb_slot_suggestion(image_rgb, dark_rail_candidate.separator_x[1:-1], suggestion_path)
    try:
        initial_lanes = detect_lanes(raw_mask, config.layout, core_box_id=core_box_id)
    except ValueError as exc:
        # A merged foreground mask must not silently become a guessed slot
        # count. Preserve the RGB proposal so a reviewer can approve or edit
        # its internal dividers, then rerun with lane_dividers_x.
        preflight_path = output_root / "lane_preflight_review.json"
        write_json(preflight_path, {
            "status": "blocked_pending_physical_slot_review",
            "reason": str(exc),
            "raw_mask_estimate": raw_lane_estimate.__dict__,
            "rgb_physical_slot_candidate": rgb_slot_candidate.to_dict() if rgb_slot_candidate else None,
            "rgb_dark_rail_candidate": dark_rail_candidate.to_dict() if dark_rail_candidate else None,
            "suggested_lane_dividers_x": (dark_rail_candidate.separator_x[1:-1]
                                          if dark_rail_candidate is not None else None),
            "suggestion_is_reviewed": False,
            "image_path": str(image_path),
            "mask_path": str(m1_2_inputs.mask_path),
        })
        write_json(output_root / "review_manifest.json", {
            "task_id": task_id,
            "needs_review": True,
            "warnings": [{"code": "physical_slot_count_unverified", "message": str(exc)}],
            "review_targets": {
                "lane_preflight_review": str(preflight_path),
                "rgb_dark_rail_suggestion": str(suggestion_path) if suggestion_path else None,
            },
        })
        raise ValueError(f"{exc}; physical-slot review proposal: {preflight_path}") from exc
    refine_result = refine_mask(raw_mask, initial_lanes, config.mask_refine)
    refined_mask = refine_result.refined_mask
    lanes = detect_lanes(refined_mask, config.layout, core_box_id=core_box_id)
    if len(lanes) != len(initial_lanes):
        raise ValueError("Lane count changed during mask refinement; review the M1-2 mask before reconstruction")

    save_mask(output_root / "refined_mask.png", refined_mask)
    save_mask(output_root / "refined_mask.tif", refined_mask)
    save_rgb(output_root / "refined_overlay.png", make_overlay(image_rgb, refined_mask))
    save_lane_debug(output_root / "lane_debug_overlay.png", image_rgb, [lane.to_dict() for lane in lanes])
    write_json(output_root / "lane_detection.json", {
        "count_source": ("reviewed_physical_dividers" if config.layout.lane_dividers_x is not None
                         else "foreground_mask_auto" if config.layout.lane_count == 0
                         else "user_configured_and_mask_checked"),
        "raw_mask_estimate": raw_lane_estimate.__dict__,
        "rgb_physical_slot_candidate": rgb_slot_candidate.to_dict() if rgb_slot_candidate else None,
        "rgb_dark_rail_candidate": dark_rail_candidate.to_dict() if dark_rail_candidate else None,
        "rgb_dark_rail_suggestion_overlay": str(suggestion_path) if suggestion_path else None,
        "reviewed_lane_dividers_x": config.layout.lane_dividers_x,
        "source_slot_prior": source_slot_prior,
        "lanes": [lane.to_dict() for lane in lanes],
    })
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
    preview_rgba = Image.fromarray(strip_result.strip_rgba, mode="RGBA")
    preview_rgba.thumbnail((1600, 4096), Image.Resampling.BILINEAR)
    preview = Image.new("RGB", preview_rgba.size, (20, 20, 20))
    preview.paste(preview_rgba, mask=preview_rgba.getchannel("A"))
    preview.save(output_root / "reconstructed_strip_preview.jpg", quality=90)
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
    if config.layout.lane_dividers_x is None and rgb_slot_candidate is not None:
        if rgb_slot_candidate.count != len(lanes):
            all_warnings.append({
                "code": "rgb_mask_slot_count_disagreement",
                "message": "RGB divider lattice and foreground mask suggest different physical slot counts; review before using reconstruction.",
                "rgb_candidate_count": rgb_slot_candidate.count,
                "mask_count": len(lanes),
                "rgb_candidate_review_required": rgb_slot_candidate.review_required,
            })
        elif rgb_slot_candidate.review_required:
            all_warnings.append({
                "code": "rgb_slot_geometry_ambiguous",
                "message": "RGB divider evidence is not decisive; physically verify the slot count before accepting reconstruction.",
                "candidate_count": rgb_slot_candidate.count,
                "score_margin": rgb_slot_candidate.score_margin,
            })
    if config.layout.lane_dividers_x is None and dark_rail_candidate is not None:
        if not dark_rail_candidate.review_required and dark_rail_candidate.count != len(lanes):
            all_warnings.append({
                "code": "dark_rail_mask_slot_count_disagreement",
                "message": "Persistent RGB tray rails disagree with the foreground-mask slot count; review physical dividers before reconstruction is used.",
                "dark_rail_count": dark_rail_candidate.count,
                "mask_count": len(lanes),
                "suggested_lane_dividers_x": dark_rail_candidate.separator_x[1:-1],
            })
    if config.layout.lane_dividers_x is not None and (
        raw_lane_estimate.reason != "ok" or raw_lane_estimate.count != len(lanes)
    ):
        all_warnings.append({
            "code": "reviewed_geometry_mask_disagreement",
            "message": "Reviewed physical slot dividers disagree with mask-only support; inspect empty slots and the foreground mask.",
            "physical_slot_count": len(lanes),
            "mask_slot_count": raw_lane_estimate.count,
            "mask_reason": raw_lane_estimate.reason,
        })
    all_warnings.extend(_physical_slot_warnings(
        source_slot_prior, len(lanes), config.layout.lane_dividers_x is not None))
    weak_lanes = [lane.lane_index for lane in lanes if lane.foreground_pixels == 0]
    if weak_lanes:
        all_warnings.append({
            "code": "empty_physical_slots",
            "message": "Physical slots with no foreground remain in the reconstruction; confirm whether they are truly empty.",
            "lane_indices": weak_lanes,
        })
    if not payload.get("depth_order_confirmed", False):
        all_warnings.append({
            "code": "depth_order_unverified",
            "message": "Lane order and direction are image conventions only; verify them against field depth labels.",
        })
    quality_report = build_quality_report(
        lane_count_expected=len(initial_lanes),
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
                "rgb_dark_rail_suggestion": str(suggestion_path) if suggestion_path else None,
            },
            "suggested_lane_dividers_x": (dark_rail_candidate.separator_x[1:-1]
                                          if dark_rail_candidate is not None else None),
            "suggestion_is_reviewed": False,
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
            "physical_slot_count_verified": config.layout.lane_dividers_x is not None,
            "source_slot_prior_count": int(source_slot_prior["physical_slot_count"]) if source_slot_prior else None,
            "segment_count": len(segments),
            "removed_component_count": refine_result.report.get("removed_component_count", 0),
            "strip_height_px": int(strip_result.strip_mask.shape[0]),
            "meters_per_pixel": depth_mapping["meters_per_pixel"],
        },
        "warnings": all_warnings,
    }
