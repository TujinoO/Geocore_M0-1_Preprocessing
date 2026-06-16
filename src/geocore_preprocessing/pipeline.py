from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classical_mask import run_classical_foreground_mask
from .fusion_accessor import FusionResultAccessor, ImageGrid
from .json_utils import read_json, write_json
from .paths import bootstrap_module_paths, find_m12_model_package


bootstrap_module_paths()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _project_value(accessor: FusionResultAccessor, key: str, default: Any = None) -> Any:
    return accessor.manifest.get("project", {}).get(key, default)


def _project_metadata_from_payload(payload: dict[str, Any]):
    from geocore_m01_fusion.config import ProjectMetadata

    return ProjectMetadata(
        project_id=payload.get("project_id"),
        borehole_id=payload.get("hole_id") or payload.get("borehole_id"),
        box_id=payload.get("box_id"),
        depth_start_m=payload.get("depth_start_m"),
        depth_end_m=payload.get("depth_end_m"),
    )


def _run_m0_if_needed(payload: dict[str, Any], output_root: Path) -> Path:
    manifest = payload.get("manifest_path") or payload.get("manifest")
    if manifest:
        return Path(manifest).resolve()

    m0_output = output_root / "m0_fusion"
    mode = payload.get("m0_mode", payload.get("mode", "classical"))
    from geocore_m01_fusion.config import EnviInput, FusionConfig
    from geocore_m01_fusion.pipeline import fuse_arrays, fuse_envi_files

    config = FusionConfig(
        mode=mode,
        streaming=bool(payload.get("m0_streaming", False)),
        export_envi=bool(payload.get("export_envi", False)),
        write_previews=not bool(payload.get("no_previews", False)),
        chunk_size=tuple(payload.get("chunk_size", (512, 512, 32))),
    )
    project = _project_metadata_from_payload(payload)
    if payload.get("m0_demo"):
        from geocore_m01_fusion.cli import make_demo_inputs

        rgb, nir, swir, nir_w, swir_w = make_demo_inputs()
        result = fuse_arrays(
            rgb,
            nir,
            swir,
            output_dir=m0_output,
            nir_wavelengths=nir_w,
            swir_wavelengths=swir_w,
            config=config,
            project=project,
            input_metadata={"source": "synthetic_demo_from_unified_preprocessing_pipeline"},
        )
        return result.output_dir / "manifest.json"

    required = ["rgb_hdr", "nir_hdr", "swir_hdr"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(
            "Either manifest_path or M0 inputs are required. "
            f"Missing M0 ENVI fields: {', '.join(missing)}"
        )
    result = fuse_envi_files(
        EnviInput(Path(payload["rgb_hdr"]), Path(payload["rgb_dat"]) if payload.get("rgb_dat") else None),
        EnviInput(Path(payload["nir_hdr"]), Path(payload["nir_dat"]) if payload.get("nir_dat") else None),
        EnviInput(Path(payload["swir_hdr"]), Path(payload["swir_dat"]) if payload.get("swir_dat") else None),
        output_dir=m0_output,
        config=config,
        project=project,
        mmap=not bool(payload.get("no_mmap", False)),
    )
    return result.output_dir / "manifest.json"


def _run_m11(accessor: FusionResultAccessor, payload: dict[str, Any], output_root: Path) -> dict[str, Any]:
    from geocore_m1_1.config import M11Config
    from geocore_m1_1.pipeline import run_m11_rotation_correction

    input_image = Path(payload.get("m1_1_input_image") or accessor.preview_path()).resolve()
    output_dir = output_root / "m1_1_rotation"
    config = M11Config(
        thumbnail_width=int(payload.get("m1_1_thumbnail_width", 512)),
        expected_box_count=payload.get("expected_box_count"),
        save_previews=not bool(payload.get("no_m1_1_previews", False)),
    )
    result = run_m11_rotation_correction(
        input_path=str(input_image),
        hdr_path=payload.get("m1_1_hdr_path"),
        output_dir=str(output_dir),
        config=config,
    )
    input_grid = _m1_input_grid(input_image, payload.get("m1_1_hdr_path"))
    metadata = result.to_dict()
    metadata["metadata_path"] = str(output_dir / "metadata.json")
    metadata["input_grid"] = input_grid.__dict__
    metadata["input_to_rgb_reference_scale"] = list(_scale_grid_to_reference(accessor, input_grid))
    return metadata


def _m1_input_grid(input_image: str | Path, hdr_path: str | None = None) -> ImageGrid:
    from geocore_m1_1.io_envi import read_input_image

    image, _ = read_input_image(input_image, hdr_path)
    return ImageGrid(width=int(image.shape[1]), height=int(image.shape[0]), name="m1_working_grid")


def _scale_grid_to_reference(accessor: FusionResultAccessor, grid: ImageGrid) -> tuple[float, float]:
    reference = accessor.grid
    return (reference.width / max(1, grid.width), reference.height / max(1, grid.height))


def _bbox_to_reference(
    accessor: FusionResultAccessor,
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    source_grid: ImageGrid,
) -> list[int]:
    sx, sy = _scale_grid_to_reference(accessor, source_grid)
    x0, y0, x1, y1 = bbox_xyxy
    reference = accessor.grid
    return [
        int(max(0, min(reference.width, round(x0 * sx)))),
        int(max(0, min(reference.height, round(y0 * sy)))),
        int(max(0, min(reference.width, round(x1 * sx)))),
        int(max(0, min(reference.height, round(y1 * sy)))),
    ]


def _build_m12_predictor(payload: dict[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    engine = str(payload.get("m1_2_engine", "auto")).lower()
    info: dict[str, Any] = {"engine_requested": engine, "engine_used": None, "model_package": None, "warnings": []}
    if engine == "classical":
        info["engine_used"] = "classical"
        return None, info

    model_package = find_m12_model_package(payload.get("m1_2_model_package"))
    if model_package is None:
        message = "M1-2 model package was not found; falling back to classical mask."
        if engine == "model":
            raise FileNotFoundError(message)
        info["warnings"].append(message)
        info["engine_used"] = "classical"
        return None, info

    try:
        from geocore_mask.inference.predictor import CoreMaskPredictor
        from geocore_mask.utils.model_package import load_model_package

        predictor = CoreMaskPredictor(load_model_package(model_package=str(model_package)))
        info["engine_used"] = "model"
        info["model_package"] = str(model_package)
        return predictor, info
    except Exception as exc:
        if engine == "model":
            raise RuntimeError(f"Failed to initialize M1-2 model package {model_package}: {exc}") from exc
        info["warnings"].append(f"M1-2 model initialization failed; falling back to classical mask: {exc}")
        info["engine_used"] = "classical"
        return None, info


def _run_m12_for_box(
    predictor: Any | None,
    box: dict[str, Any],
    payload: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    image_path = box["output_image"]
    output_dir = output_root / "m1_2_foreground_mask" / box["box_id"]
    if predictor is None:
        result = run_classical_foreground_mask(
            image_path,
            output_dir,
            threshold=payload.get("m1_2_threshold"),
            keep_largest_component=bool(payload.get("m1_2_keep_largest_component", False)),
        )
    else:
        result = predictor.predict(
            image_path,
            output_dir,
            threshold=payload.get("m1_2_threshold"),
            enable_postprocess=payload.get("m1_2_enable_postprocess"),
            output_preview=True,
        )
    result["output_dir"] = str(output_dir)
    result["input_image"] = image_path
    return result


def _resolve_depth_range(
    accessor: FusionResultAccessor,
    payload: dict[str, Any],
    box_count: int,
    warnings: list[str],
) -> tuple[float, float]:
    start = payload.get("depth_start_m", _project_value(accessor, "depth_start_m"))
    end = payload.get("depth_end_m", _project_value(accessor, "depth_end_m"))
    if start is None or end is None:
        warnings.append("depth_start_m/depth_end_m were not provided; using synthetic 1 m per detected box.")
        return 0.0, float(max(1, box_count))
    return float(start), float(end)


def _box_depth_window(index: int, count: int, depth_start: float, depth_end: float) -> tuple[float, float]:
    count = max(1, count)
    span = depth_end - depth_start
    box_start = depth_start + span * index / count
    box_end = depth_start + span * (index + 1) / count
    return float(box_start), float(box_end)


def _run_m13_for_box(
    box: dict[str, Any],
    m12_result: dict[str, Any],
    payload: dict[str, Any],
    output_root: Path,
    *,
    hole_id: str,
    core_box_id: str,
    depth_start_m: float,
    depth_end_m: float,
) -> dict[str, Any]:
    from geocore_m1_3.pipeline import run_segment_depth

    m13_payload: dict[str, Any] = {
        "task_id": core_box_id,
        "image_path": box["output_image"],
        "m1_2_output_dir": m12_result["output_dir"],
        "output_dir": str(output_root / "m1_3_separator_mark"),
        "hole_id": hole_id,
        "core_box_id": core_box_id,
        "depth_start_m": depth_start_m,
        "depth_end_m": depth_end_m,
        "segmentation": {
            "segment_length_cm": float(payload.get("segment_length_cm", 10.0)),
            "overlap_cm": float(payload.get("overlap_cm", 0.0)),
        },
    }
    if payload.get("lane_count") is not None:
        m13_payload["layout"] = {"lane_count": int(payload["lane_count"])}
    if payload.get("lane_order"):
        m13_payload.setdefault("layout", {})["lane_order"] = payload["lane_order"]
    if payload.get("lane_direction"):
        m13_payload.setdefault("layout", {})["lane_direction"] = payload["lane_direction"]
    if payload.get("missing_intervals"):
        m13_payload["missing_intervals"] = payload["missing_intervals"]
    if payload.get("depth_anchors"):
        m13_payload["depth_anchors"] = payload["depth_anchors"]
    if payload.get("gap_policy"):
        m13_payload["gap_policy"] = payload["gap_policy"]
    return run_segment_depth(m13_payload)


def _estimate_corrected_bbox_from_strip(source_strip_bbox: list[int], strip_mapping: dict[str, Any]) -> list[int] | None:
    _, y0, _, y1 = [int(v) for v in source_strip_bbox]
    rects: list[list[float]] = []
    for lane in strip_mapping.get("lanes", []):
        ly0 = int(lane.get("strip_y_start", 0))
        ly1 = int(lane.get("strip_y_end", 0))
        overlap0 = max(y0, ly0)
        overlap1 = min(y1, ly1)
        if overlap1 <= overlap0:
            continue
        sx0, sy0, sx1, sy1 = [float(v) for v in lane.get("source_bbox", [0, 0, 0, 0])]
        denom = max(1.0, float(ly1 - ly0))
        t0 = (overlap0 - ly0) / denom
        t1 = (overlap1 - ly0) / denom
        rects.append([sx0, sy0 + t0 * (sy1 - sy0), sx1, sy0 + t1 * (sy1 - sy0)])
    if not rects:
        return None
    return [
        int(min(rect[0] for rect in rects)),
        int(min(rect[1] for rect in rects)),
        int(max(rect[2] for rect in rects)),
        int(max(rect[3] for rect in rects)),
    ]


def _augment_segments_with_cube_refs(
    accessor: FusionResultAccessor,
    m13_result: dict[str, Any],
    box_context: dict[str, Any],
    m12_result: dict[str, Any],
) -> Path:
    segments_path = Path(m13_result["output_files"]["segments_json"])
    output_dir = Path(m13_result["output_dir"])
    segments_doc = read_json(segments_path)
    strip_mapping_path = output_dir / "strip_mapping.json"
    strip_mapping = read_json(strip_mapping_path) if strip_mapping_path.exists() else {}
    mineral_band_count = 0
    try:
        mineral_band_count = len(accessor.selected_band_indices("mineral"))
    except Exception:
        mineral_band_count = 0
    for segment in segments_doc.get("segments", []):
        segment["grid"] = "reconstructed_strip"
        segment["cube_ref"] = accessor.cube_ref()
        segment["transform_stack_ref"] = box_context.get("transform_stack_ref")
        segment["source_refs"] = {
            "m0_manifest": str(accessor.manifest_path),
            "m1_1_box_id": box_context["box_id"],
            "m1_1_corrected_image": box_context["corrected_image"],
            "m1_2_output_dir": m12_result["output_dir"],
            "m1_2_mask_path": m12_result["output_files"].get("mask_png"),
        }
        segment["bbox_corrected_box_estimate"] = _estimate_corrected_bbox_from_strip(
            segment.get("source_strip_bbox", [0, 0, 0, 0]),
            strip_mapping,
        )
        segment["bbox_rgb_reference_parent_box"] = box_context.get("bbox_xyxy_rgb_reference")
        segment["bbox_mapping_status"] = (
            "segment is exact in reconstructed_strip; bbox_rgb_reference_parent_box is a parent "
            "M1-1 box reference for reading candidate Zarr windows."
        )
        segment["band_selection"] = {
            "recommended_mineral_band_count": mineral_band_count,
            "band_metadata": str(accessor.band_metadata_path),
        }
    out_path = output_dir / "segments_with_cube_refs.json"
    write_json(out_path, segments_doc)
    return out_path


def run_preprocessing_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """Run M0-1 -> M1-1 -> M1-2 -> M1-3 and write a linked preprocessing context."""

    output_root = Path(payload["output_dir"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    manifest_path = _run_m0_if_needed(payload, output_root)
    accessor = FusionResultAccessor(manifest_path)
    preview_path = Path(payload.get("m1_1_input_image") or accessor.preview_path()).resolve()
    m11_metadata = _run_m11(accessor, payload, output_root)
    m1_input_grid = ImageGrid(**m11_metadata["input_grid"])
    boxes = sorted(m11_metadata.get("boxes", []), key=lambda item: int(item.get("order_index", 0)))
    if payload.get("limit_boxes"):
        boxes = boxes[: int(payload["limit_boxes"])]
    if not boxes:
        raise RuntimeError("M1-1 did not detect any core boxes; cannot continue to M1-2/M1-3.")

    depth_start, depth_end = _resolve_depth_range(accessor, payload, len(boxes), warnings)
    hole_id = str(payload.get("hole_id") or _project_value(accessor, "borehole_id", "UNKNOWN_HOLE"))
    box_prefix = str(payload.get("core_box_prefix") or _project_value(accessor, "box_id", "CORE_BOX"))
    predictor, m12_engine_info = _build_m12_predictor(payload)
    warnings.extend(m12_engine_info.get("warnings", []))

    transform_stack_path = output_root / "m1_transform_stack.json"
    context: dict[str, Any] = {
        "schema_version": "geocore_preprocessing_context.v1",
        "created_at": _now(),
        "output_dir": str(output_root),
        "fusion": {
            "manifest_path": str(accessor.manifest_path),
            "grid": accessor.grid.__dict__,
            "preview_path": str(preview_path),
            "cube_ref": accessor.cube_ref(),
            "quality_report": str(accessor.quality_report_path) if accessor.quality_report_path else None,
        },
        "m1_1": {
            "output_dir": str(output_root / "m1_1_rotation"),
            "metadata_path": m11_metadata.get("metadata_path"),
            "input_image": str(preview_path),
            "input_grid": m11_metadata.get("input_grid"),
            "input_to_rgb_reference_scale": m11_metadata.get("input_to_rgb_reference_scale"),
            "box_count": len(boxes),
        },
        "m1_2": {
            "engine": m12_engine_info,
            "output_dir": str(output_root / "m1_2_foreground_mask"),
        },
        "m1_3": {
            "output_dir": str(output_root / "m1_3_separator_mark"),
            "depth_start_m": depth_start,
            "depth_end_m": depth_end,
        },
        "boxes": [],
        "warnings": warnings,
    }

    transforms: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        box_id = str(box["box_id"])
        rgb_bbox = _bbox_to_reference(accessor, box["bbox_xyxy_raw"], m1_input_grid)
        d0, d1 = _box_depth_window(index, len(boxes), depth_start, depth_end)
        core_box_id = f"{box_prefix}_{box_id}"
        box_context = {
            "box_id": box_id,
            "core_box_id": core_box_id,
            "order_index": box.get("order_index"),
            "depth_start_m": d0,
            "depth_end_m": d1,
            "bbox_xyxy_m1_input": box.get("bbox_xyxy_raw"),
            "bbox_xyxy_rgb_reference": rgb_bbox,
            "angle_deg": box.get("angle_deg"),
            "confidence": box.get("confidence"),
            "corrected_image": box.get("output_image"),
            "m1_1_mask": box.get("output_mask"),
            "transform_stack_ref": str(transform_stack_path),
        }
        transforms.append(
            {
                "name": f"m1_1_{box_id}_rotation",
                "type": "crop_rotate_nearest",
                "source_grid": "m1_working_grid",
                "target_grid": f"corrected_box:{box_id}",
                "bbox_xyxy_m1_input": box.get("bbox_xyxy_raw"),
                "bbox_xyxy_rgb_reference": rgb_bbox,
                "rotation_matrix_2x3": box.get("rotation_matrix_2x3"),
                "angle_deg": box.get("angle_deg"),
                "corrected_image": box.get("output_image"),
                "corrected_mask": box.get("output_mask"),
            }
        )

        m12_result = _run_m12_for_box(predictor, box, payload, output_root)
        m13_result = _run_m13_for_box(
            box,
            m12_result,
            payload,
            output_root,
            hole_id=hole_id,
            core_box_id=core_box_id,
            depth_start_m=d0,
            depth_end_m=d1,
        )
        segments_with_refs = _augment_segments_with_cube_refs(accessor, m13_result, box_context, m12_result)
        box_context["m1_2"] = m12_result
        box_context["m1_3"] = m13_result
        box_context["segments_with_cube_refs"] = str(segments_with_refs)
        context["boxes"].append(box_context)

    transform_stack = {
        "schema_version": "geocore_m1_transform_stack.v1",
        "created_at": _now(),
        "reference_manifest": str(accessor.manifest_path),
        "reference_grid": accessor.grid.__dict__,
        "m1_working_image": str(preview_path),
        "m1_working_grid": m1_input_grid.__dict__,
        "transforms": transforms,
    }
    write_json(transform_stack_path, transform_stack)
    context["m1_1"]["transform_stack"] = str(transform_stack_path)

    context_path = output_root / "preprocess_context.json"
    result_path = output_root / "pipeline_result.json"
    write_json(context_path, context)
    result = {
        "status": "succeeded",
        "created_at": _now(),
        "output_dir": str(output_root),
        "output_files": {
            "preprocess_context": str(context_path),
            "pipeline_result": str(result_path),
            "m0_manifest": str(accessor.manifest_path),
            "m1_transform_stack": str(transform_stack_path),
        },
        "metrics": {
            "box_count": len(context["boxes"]),
            "segment_count": sum(int(box["m1_3"]["metrics"]["segment_count"]) for box in context["boxes"]),
        },
        "warnings": context["warnings"],
    }
    write_json(result_path, result)
    return result


def inspect_manifest(manifest_path: str | Path) -> dict[str, Any]:
    accessor = FusionResultAccessor(manifest_path)
    preview = accessor.preview_path()
    preview_grid = accessor.preview_grid(preview)
    return {
        "manifest_path": str(accessor.manifest_path),
        "root": str(accessor.root),
        "grid": accessor.grid.__dict__,
        "preview_path": str(preview),
        "preview_grid": preview_grid.__dict__,
        "preview_to_reference_scale": list(accessor.scale_from_preview_to_reference(preview)),
        "cube_path": str(accessor.cube_path),
        "band_metadata": str(accessor.band_metadata_path),
        "registration_model": str(accessor.registration_model_path) if accessor.registration_model_path else None,
        "quality_report": str(accessor.quality_report_path) if accessor.quality_report_path else None,
    }
