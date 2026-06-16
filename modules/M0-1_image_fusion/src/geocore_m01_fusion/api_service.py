"""Framework-neutral service functions for M0-1 backend integration.

The functions in this file are intentionally plain Python.  A Django, FastAPI,
Flask, or desktop backend can call them from HTTP handlers without coupling the
algorithm package to a specific web framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EnviInput, FusionConfig, FusionMode, ProjectMetadata
from .envi import read_envi, write_envi
from .pipeline import fuse_envi_files
from .registration import LocalWarpModel, TiePoint, register_local_warp
from .resample import resize_cube
from .roi import _joint_hsi_structure, _rgb_gray_resized, prepare_aligned_roi
from .tiepoints import (
    apply_tie_point_operations,
    build_warp_model_from_session,
    create_session_from_model,
    read_session,
    render_tie_point_context_visualization,
    render_tie_point_visualization,
    session_quality,
    write_session,
)


STAGE_PRESETS: dict[str, dict[str, str]] = {
    "swir_to_rgb_lowres": {
        "reference_name": "RGB low-resolution structure",
        "moving_name": "SWIR structure",
        "description": "SWIR local warp to the RGB low-resolution reference grid.",
    },
    "nir_to_swir_lowres": {
        "reference_name": "SWIR structure",
        "moving_name": "NIR structure",
        "description": "NIR local warp to the already-registered SWIR grid.",
    },
    "joint_hsi_to_rgb_lowres_final": {
        "reference_name": "RGB low-resolution structure",
        "moving_name": "Joint NIR+SWIR HSI structure",
        "description": "Final joint HSI-to-RGB fine warp applied to both NIR and SWIR.",
    },
}


def create_tie_point_session_from_roi(
    roi_dir: str | Path,
    *,
    stage: str,
    output_dir: str | Path | None = None,
    render_preview: bool = True,
) -> dict[str, Any]:
    """Create an editable tie-point session from an existing ROI manifest."""

    roi = Path(roi_dir)
    manifest_path = roi / "roi_manifest.json"
    manifest = _read_json(manifest_path)
    model_data = manifest.get("warp_models", {}).get(stage)
    if not model_data:
        raise KeyError(f"Warp model stage not found in ROI manifest: {stage}")
    preset = STAGE_PRESETS.get(stage, {"reference_name": "reference", "moving_name": "moving", "description": stage})
    model = _local_warp_model_from_dict(model_data)
    session = create_session_from_model(
        model,
        stage=stage,
        reference_name=preset["reference_name"],
        moving_name=preset["moving_name"],
        metadata={
            "roi_dir": str(roi),
            "roi_manifest": str(manifest_path),
            "stage_description": preset["description"],
            "source": "roi_manifest",
        },
    )
    out = Path(output_dir) if output_dir is not None else roi / "tiepoint_sessions" / stage
    session_path = write_session(session, out / "session.json")
    response: dict[str, Any] = {
        "session_path": str(session_path),
        "session": session.to_dict(),
        "preview_paths": {},
    }
    if render_preview:
        reference, moving = load_stage_preview_images(roi, stage)
        response["preview_paths"] = render_tie_point_visualization(reference, moving, session, out / "previews")
        reference_context, moving_context = load_stage_context_images(roi, stage)
        response["preview_paths"].update(
            render_tie_point_context_visualization(reference_context, moving_context, session, out / "previews")
        )
    return response


def create_tie_point_session_from_arrays(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    stage: str,
    output_dir: str | Path,
    reference_name: str = "reference",
    moving_name: str = "moving",
    registration_params: dict[str, Any] | None = None,
    render_preview: bool = True,
) -> dict[str, Any]:
    """Run automatic tie-point matching and save an editable session."""

    params = dict(registration_params or {})
    model = register_local_warp(reference, moving, **params)
    session = create_session_from_model(
        model,
        stage=stage,
        reference_name=reference_name,
        moving_name=moving_name,
        metadata={
            "source": "arrays",
            "registration_params": params,
        },
    )
    out = Path(output_dir)
    session_path = write_session(session, out / "session.json")
    preview_paths: dict[str, str] = {}
    if render_preview:
        preview_paths = render_tie_point_visualization(reference, moving, session, out / "previews")
    return {
        "session_path": str(session_path),
        "session": session.to_dict(),
        "preview_paths": preview_paths,
        "model": model.to_dict(),
    }


def prepare_aligned_roi_job(
    *,
    root: str | Path,
    output_dir: str | Path,
    crop_height: int = 768,
    crop_width: int = 512,
    preview_width: int = 256,
    max_preview_height: int = 3072,
    registration_mode: str = "scale_only",
    local_refine: bool = True,
    anchor_nir_to_swir: bool = True,
    refine_anchored_nir: bool = True,
    local_warp: bool = True,
    final_hsi_rgb_warp: bool = True,
) -> dict[str, Any]:
    """API-friendly wrapper around the current ROI registration pipeline."""

    return prepare_aligned_roi(
        root,
        output_dir,
        crop_height=crop_height,
        crop_width=crop_width,
        preview_width=preview_width,
        max_preview_height=max_preview_height,
        registration_mode=registration_mode,
        local_refine=local_refine,
        anchor_nir_to_swir=anchor_nir_to_swir,
        refine_anchored_nir=refine_anchored_nir,
        local_warp=local_warp,
        final_hsi_rgb_warp=final_hsi_rgb_warp,
    )


def update_tie_point_session(
    session_path: str | Path,
    operations: list[dict[str, Any]],
    *,
    roi_dir: str | Path | None = None,
    render_preview: bool = True,
) -> dict[str, Any]:
    """Apply frontend edit operations and optionally refresh preview images."""

    session = read_session(session_path)
    edit_result = apply_tie_point_operations(session, operations)
    out_path = write_session(session, session_path)
    preview_paths: dict[str, str] = {}
    roi = Path(roi_dir) if roi_dir is not None else _roi_dir_from_session(session.to_dict())
    if render_preview and roi is not None:
        reference, moving = load_stage_preview_images(roi, session.stage)
        preview_paths = render_tie_point_visualization(reference, moving, session, out_path.parent / "previews")
        reference_context, moving_context = load_stage_context_images(roi, session.stage)
        preview_paths.update(
            render_tie_point_context_visualization(reference_context, moving_context, session, out_path.parent / "previews")
        )
    return {
        "session_path": str(out_path),
        "changed_point_ids": edit_result["changed_point_ids"],
        "quality": session_quality(session),
        "preview_paths": preview_paths,
        "session": session.to_dict(),
    }


def apply_tie_point_session_to_envi(
    session_path: str | Path,
    *,
    input_hdr: str | Path,
    input_dat: str | Path | None,
    output_hdr: str | Path,
    output_dat: str | Path | None = None,
    min_active_points: int = 3,
) -> dict[str, Any]:
    """Warp one ENVI cube using the currently active tie points."""

    session = read_session(session_path)
    cube, meta = read_envi(input_hdr, input_dat, mmap=True)
    model = build_warp_model_from_session(session, min_active_points=min_active_points)
    warped = _warp_in_blocks(np.asarray(cube), model)
    hdr, dat = write_envi(
        warped.astype(np.float32, copy=False),
        output_hdr,
        output_dat,
        wavelengths=meta.wavelengths,
        interleave="bil",
        description=f"Interactive tie-point warped {session.stage}",
    )
    model_path = Path(hdr).with_suffix(".registration_model.json")
    model_path.write_text(json.dumps(_json_safe(model.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_hdr": str(hdr),
        "output_dat": str(dat),
        "registration_model": str(model_path),
        "quality": session_quality(session),
        "model": model.to_dict(),
    }


def run_fusion_job(
    *,
    rgb_hdr: str | Path,
    rgb_dat: str | Path | None,
    nir_hdr: str | Path,
    nir_dat: str | Path | None,
    swir_hdr: str | Path,
    swir_dat: str | Path | None,
    output_dir: str | Path,
    mode: str = FusionMode.CLASSICAL.value,
    streaming: bool = True,
    chunk_size: tuple[int, int, int] = (256, 256, 32),
    export_envi: bool = False,
    write_previews: bool = True,
    fast_detail_strength: float | None = None,
    classical_detail_strength: float | None = None,
    deep_detail_strength: float | None = None,
    spatial_detail_strength: float | None = None,
    spatial_detail_small_radius: int | None = None,
    spatial_detail_large_radius: int | None = None,
    project: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the existing fusion pipeline from API-friendly primitive values."""

    config = FusionConfig(
        mode=mode,
        streaming=streaming,
        chunk_size=chunk_size,
        export_envi=export_envi,
        write_previews=write_previews,
    )
    if fast_detail_strength is not None:
        config.fast_detail_strength = float(fast_detail_strength)
    if classical_detail_strength is not None:
        config.classical_detail_strength = float(classical_detail_strength)
    if deep_detail_strength is not None:
        config.deep_detail_strength = float(deep_detail_strength)
    if spatial_detail_strength is not None:
        config.spatial_detail_strength = float(spatial_detail_strength)
    if spatial_detail_small_radius is not None:
        config.spatial_detail_small_radius = int(spatial_detail_small_radius)
    if spatial_detail_large_radius is not None:
        config.spatial_detail_large_radius = int(spatial_detail_large_radius)
    project_meta = None
    if project:
        project_meta = ProjectMetadata(
            project_id=project.get("project_id"),
            borehole_id=project.get("borehole_id"),
            box_id=project.get("box_id"),
            depth_start_m=project.get("depth_start_m"),
            depth_end_m=project.get("depth_end_m"),
        )
    result = fuse_envi_files(
        EnviInput(Path(rgb_hdr), Path(rgb_dat) if rgb_dat else None),
        EnviInput(Path(nir_hdr), Path(nir_dat) if nir_dat else None),
        EnviInput(Path(swir_hdr), Path(swir_dat) if swir_dat else None),
        output_dir=output_dir,
        config=config,
        project=project_meta,
        mmap=True,
    )
    return {
        "output_dir": str(result.output_dir),
        "manifest": result.manifest,
        "quality_report": result.quality_report,
    }


def load_stage_preview_images(roi_dir: str | Path, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Load reference/moving structure images for frontend tie-point previews."""

    roi = Path(roi_dir)
    rgb, _ = read_envi(roi / "aligned_envi" / "RGB_aligned_roi.hdr", roi / "aligned_envi" / "RGB_aligned_roi.dat", mmap=True)
    nir, _ = read_envi(roi / "aligned_envi" / "NIR_aligned_roi.hdr", roi / "aligned_envi" / "NIR_aligned_roi.dat", mmap=True)
    swir, _ = read_envi(roi / "aligned_envi" / "SWIR_aligned_roi.hdr", roi / "aligned_envi" / "SWIR_aligned_roi.dat", mmap=True)
    nir_struct = _hsi_mean(nir)
    swir_struct = _hsi_mean(swir)
    if stage == "swir_to_rgb_lowres":
        reference = _rgb_gray_resized(np.asarray(rgb[:, :, :3]), swir_struct.shape)
        return reference, swir_struct
    if stage == "nir_to_swir_lowres":
        if nir_struct.shape != swir_struct.shape:
            nir_struct = resize_cube(nir_struct, swir_struct.shape, method="bilinear")
        return swir_struct, nir_struct
    if stage == "joint_hsi_to_rgb_lowres_final":
        moving = _joint_hsi_structure(np.asarray(nir), np.asarray(swir))
        reference = _rgb_gray_resized(np.asarray(rgb[:, :, :3]), moving.shape)
        return reference, moving
    raise ValueError(f"Unsupported tie-point stage: {stage}")


def load_stage_context_images(roi_dir: str | Path, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Load human-readable RGB/false-color context images for tie-point previews."""

    roi = Path(roi_dir)
    rgb, _ = read_envi(roi / "aligned_envi" / "RGB_aligned_roi.hdr", roi / "aligned_envi" / "RGB_aligned_roi.dat", mmap=True)
    nir, _ = read_envi(roi / "aligned_envi" / "NIR_aligned_roi.hdr", roi / "aligned_envi" / "NIR_aligned_roi.dat", mmap=True)
    swir, _ = read_envi(roi / "aligned_envi" / "SWIR_aligned_roi.hdr", roi / "aligned_envi" / "SWIR_aligned_roi.dat", mmap=True)
    rgb_context = np.asarray(rgb[:, :, :3])
    nir_context = resize_cube(_hsi_false_color(nir), rgb_context.shape[:2], method="bilinear")
    swir_context = resize_cube(_hsi_false_color(swir), rgb_context.shape[:2], method="bilinear")
    if stage == "swir_to_rgb_lowres":
        return rgb_context, swir_context
    if stage == "nir_to_swir_lowres":
        return swir_context, nir_context
    if stage == "joint_hsi_to_rgb_lowres_final":
        joint = _normalize_rgb_float(0.45 * nir_context.astype(np.float32) + 0.55 * swir_context.astype(np.float32))
        return rgb_context, joint
    raise ValueError(f"Unsupported tie-point stage: {stage}")


def available_api_operations() -> dict[str, Any]:
    """Return a compact machine-readable API summary for backend routing."""

    return {
        "module": "M0-1",
        "api_version": "m0_fusion_api.v1",
        "workflows": [
            "prepare_aligned_roi",
            "create_tie_point_session",
            "edit_tie_point_session",
            "apply_tie_point_session_to_envi",
            "run_fusion",
        ],
        "tie_point_stages": STAGE_PRESETS,
        "tie_point_operations": ["add", "update", "reject", "delete", "activate", "restore"],
        "fusion_modes": [mode.value for mode in FusionMode],
        "recommended_chunk_size": [256, 256, 32],
    }


def _local_warp_model_from_dict(data: dict[str, Any]) -> LocalWarpModel:
    return LocalWarpModel(
        reference_shape=tuple(int(x) for x in data["reference_shape"]),
        moving_shape=tuple(int(x) for x in data["moving_shape"]),
        approx_offset_y=float(data.get("approx_offset_y", 0.0)),
        approx_offset_x=float(data.get("approx_offset_x", 0.0)),
        approx_scale_y=float(data.get("approx_scale_y", 1.0)),
        approx_scale_x=float(data.get("approx_scale_x", 1.0)),
        tie_points=[_tie_point_from_dict(point) for point in data.get("tie_points", [])],
        rejected_tie_points=[_tie_point_from_dict(point) for point in data.get("rejected_tie_points", [])],
        method=str(data.get("method", "grid_tie_points_idw_warp")),
    )


def _tie_point_from_dict(data: dict[str, Any]) -> TiePoint:
    return TiePoint(
        ref_y=float(data["ref_y"]),
        ref_x=float(data["ref_x"]),
        moving_y=float(data["moving_y"]),
        moving_x=float(data["moving_x"]),
        approx_y=float(data.get("approx_y", data["moving_y"])),
        approx_x=float(data.get("approx_x", data["moving_x"])),
        score=float(data.get("score", 1.0)),
    )


def _warp_in_blocks(cube: np.ndarray, model: LocalWarpModel, *, band_block: int = 64) -> np.ndarray:
    """Warp a cube in spectral blocks to limit peak memory during API calls."""

    bands = cube.shape[2]
    blocks = []
    from .registration import warp_cube_with_model

    for b0 in range(0, bands, band_block):
        b1 = min(b0 + band_block, bands)
        blocks.append(warp_cube_with_model(cube[:, :, b0:b1], model))
    return np.concatenate(blocks, axis=2).astype(np.float32)


def _hsi_mean(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    img = np.nanmean(arr, axis=2)
    valid = np.isfinite(img)
    if not valid.any():
        return np.zeros(img.shape, dtype=np.float32)
    lo, hi = np.percentile(img[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _hsi_false_color(cube: np.ndarray) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] == 0:
        raise ValueError("HSI context image expects a y,x,band cube")
    bands = [0, arr.shape[2] // 2, arr.shape[2] - 1]
    channels = [_stretch(arr[:, :, band]) for band in bands]
    return np.stack(channels, axis=2).astype(np.uint8)


def _stretch(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255).round().astype(np.uint8)


def _normalize_rgb_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.uint8)
    for channel in range(arr.shape[2]):
        out[:, :, channel] = _stretch(arr[:, :, channel])
    return out


def _roi_dir_from_session(data: dict[str, Any]) -> Path | None:
    raw = data.get("metadata", {}).get("roi_dir")
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
