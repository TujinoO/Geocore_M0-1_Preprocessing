"""Out-of-core streaming backend for full-size ENVI fusion jobs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .algorithms import (
    _inject_rgb_detail,
    _inject_rgb_spatial_detail,
    _project_onto_basis,
    _reconstruct_low_rank,
    _sanitize_cube,
    _stable_pca_basis,
)
from .config import EnviInput, FusionConfig, FusionMode, ProjectMetadata
from .envi import read_envi
from .metrics import edge_correlation
from .output import (
    ZarrV2ChunkWriter,
    write_band_metadata,
    write_json,
)
from .preprocess import normalize_rgb
from .spectral import calibrate_and_concat, default_wavelengths


@dataclass(slots=True)
class StreamingFusionRunResult:
    """Result object for streaming jobs that do not keep the full cube in memory."""

    manifest: dict[str, Any]
    quality_report: dict[str, Any]
    output_dir: Path
    product: None = None


def fuse_envi_files_streaming(
    rgb: EnviInput,
    nir: EnviInput,
    swir: EnviInput,
    *,
    output_dir: str | Path,
    config: FusionConfig,
    project: ProjectMetadata | None = None,
    mmap: bool = True,
) -> StreamingFusionRunResult:
    """Fuse full-size ENVI inputs without materializing the full output cube."""

    mode = config.normalized_mode()
    out = Path(output_dir)
    _prepare_output_dirs(out)

    rgb_arr, rgb_meta = read_envi(rgb.hdr_path, rgb.dat_path, mmap=mmap)
    nir_arr, nir_meta = read_envi(nir.hdr_path, nir.dat_path, mmap=mmap)
    swir_arr, swir_meta = read_envi(swir.hdr_path, swir.dat_path, mmap=mmap)
    target_shape = rgb_arr.shape[:2]
    nir_w = nir_meta.wavelengths or default_wavelengths(nir_arr.shape[2])
    swir_w = swir_meta.wavelengths or default_wavelengths(swir_arr.shape[2])

    sample_nir, sample_swir = _sample_paired_spectra(
        nir_arr,
        swir_arr,
        max_samples=config.max_basis_pixels,
        seed=config.random_seed,
    )
    nir_scale = _estimate_reflectance_scale(sample_nir)
    swir_scale = _estimate_reflectance_scale(sample_swir)
    sample_nir = sample_nir / nir_scale
    sample_swir = sample_swir / swir_scale
    stitch = calibrate_and_concat(
        sample_nir[:, None, :],
        sample_swir[:, None, :],
        nir_w,
        swir_w,
        mask=None,
    )
    swir_gain = float(stitch.stitch_model.get("swir_gain", 1.0))
    swir_offset = float(stitch.stitch_model.get("swir_offset", 0.0))
    sample_concat = np.concatenate(
        [sample_nir, sample_swir * swir_gain + swir_offset],
        axis=1,
    ).astype(np.float32)
    mean, basis = _fit_global_basis(
        sample_concat,
        rank=config.rank,
        seed=config.random_seed,
    )

    cube_shape = (int(target_shape[0]), int(target_shape[1]), int(sample_concat.shape[1]))
    writer = ZarrV2ChunkWriter(
        out / "fused_cube.zarr",
        shape=cube_shape,
        chunks=config.chunk_size,
        dtype=config.output_dtype,
        attrs={
            "array_name": "reflectance",
            "axis_order": "y,x,band",
            "unit": "reflectance",
            "algorithm_mode": mode.value,
            "backend": "streaming",
        },
    )

    chunk_y, chunk_x, _ = config.chunk_size
    blocks_processed = 0
    edge_scores: list[float] = []
    min_value = float("inf")
    max_value = float("-inf")
    for y0 in range(0, target_shape[0], chunk_y):
        y1 = min(y0 + chunk_y, target_shape[0])
        for x0 in range(0, target_shape[1], chunk_x):
            x1 = min(x0 + chunk_x, target_shape[1])
            margin = _tile_margin(mode, config)
            ey0 = max(0, y0 - margin)
            ey1 = min(target_shape[0], y1 + margin)
            ex0 = max(0, x0 - margin)
            ex1 = min(target_shape[1], x1 + margin)
            rgb_tile = np.asarray(rgb_arr[ey0:ey1, ex0:ex1, :3])
            base_tile = _build_base_tile(
                nir_arr,
                swir_arr,
                target_shape,
                (ey0, ey1, ex0, ex1),
                nir_scale=nir_scale,
                swir_scale=swir_scale,
                swir_gain=swir_gain,
                swir_offset=swir_offset,
                method=config.interpolation,
            )
            fused_ext = _fuse_tile(base_tile, rgb_tile, mode, config, mean, basis)
            fused_tile = fused_ext[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0, :]
            writer.write_tile(y0, x0, fused_tile)
            blocks_processed += 1
            min_value = min(min_value, float(np.nanmin(fused_tile)))
            max_value = max(max_value, float(np.nanmax(fused_tile)))
            if len(edge_scores) < 64:
                rgb_center = rgb_tile[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0, :]
                edge_scores.append(edge_correlation(fused_tile, rgb_center))

    band_metadata_path = out / "metadata" / "band_metadata.csv"
    write_band_metadata(stitch.band_metadata, band_metadata_path)
    write_json(stitch.stitch_model, out / "metadata" / "spectral_stitch_model.json")
    input_metadata = {
        "rgb": _meta_to_dict(rgb_meta, rgb),
        "nir": _meta_to_dict(nir_meta, nir),
        "swir": _meta_to_dict(swir_meta, swir),
    }
    write_json(input_metadata, out / "metadata" / "input_metadata.json")
    write_json(_config_to_dict(config), out / "metadata" / "processing_config.json")
    write_json(_registration_model(target_shape), out / "metadata" / "registration_model.json")

    preview_paths = {}
    if config.write_previews:
        preview_paths = _write_streaming_previews(rgb_arr, out / "previews", config.streaming_preview_max_size)

    quality_report = {
        "summary": {
            "status": "completed",
            "notes": "Streaming metrics are lightweight because the full fused cube is never materialized in memory.",
        },
        "streaming": {
            "blocks_processed": blocks_processed,
            "chunk_size": [int(x) for x in config.chunk_size],
            "backend": "streaming_zarr_v2",
            "output_min": min_value,
            "output_max": max_value,
        },
        "spectral": {
            "swir_gain": swir_gain,
            "swir_offset": swir_offset,
            "nir_scale": nir_scale,
            "swir_scale": swir_scale,
            "overlap_pair_count": stitch.stitch_model.get("overlap_pair_count", 0),
        },
        "spatial": {
            "sample_rgb_edge_correlation_mean": _safe_mean(edge_scores),
        },
        "algorithm": _algorithm_details(mode, config, basis),
        "warnings": _streaming_warnings(config),
    }
    write_json(quality_report, out / "metrics" / "quality_report.json")

    manifest = _manifest(
        out,
        mode,
        config,
        project,
        cube_shape,
        preview_paths,
        export_envi_available=False,
    )
    write_json(manifest, out / "manifest.json")
    return StreamingFusionRunResult(manifest=manifest, quality_report=quality_report, output_dir=out)


def _build_base_tile(
    nir_arr: np.ndarray,
    swir_arr: np.ndarray,
    target_shape: tuple[int, int],
    window: tuple[int, int, int, int],
    *,
    nir_scale: float,
    swir_scale: float,
    swir_gain: float,
    swir_offset: float,
    method: str,
) -> np.ndarray:
    y0, y1, x0, x1 = window
    nir_tile = _resample_source_window(nir_arr, target_shape, y0, y1, x0, x1, method) / nir_scale
    swir_tile = _resample_source_window(swir_arr, target_shape, y0, y1, x0, x1, method) / swir_scale
    swir_tile = swir_tile * swir_gain + swir_offset
    return np.concatenate([nir_tile, swir_tile], axis=2).astype(np.float32, copy=False)


def _fuse_tile(
    base_tile: np.ndarray,
    rgb_tile: np.ndarray,
    mode: FusionMode,
    config: FusionConfig,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    rgb01 = normalize_rgb(rgb_tile)
    if mode == FusionMode.UPSAMPLE_ONLY:
        return _sanitize_cube(base_tile)
    if mode == FusionMode.FAST_PREVIEW:
        out = _inject_rgb_detail(
            base_tile,
            rgb01,
            strength=config.fast_detail_strength,
            mask=None,
            relative=True,
        )
        out = _inject_rgb_spatial_detail(
            out,
            rgb01,
            strength=config.spatial_detail_strength * 0.9,
            mask=None,
            small_radius=config.spatial_detail_small_radius,
            large_radius=config.spatial_detail_large_radius,
        )
        return _sanitize_cube(out)
    if mode == FusionMode.CLASSICAL:
        return _classical_tile(base_tile, rgb01, config, mean, basis)
    if mode == FusionMode.DEEP_UNSUPERVISED:
        current = base_tile
        for step in range(max(1, config.deep_iterations)):
            current = _classical_tile(
                current,
                rgb01,
                config,
                mean,
                basis,
                strength=config.deep_detail_strength / float(step + 1),
                spatial_strength=config.spatial_detail_strength * (0.75 + 0.25 / float(step + 1)),
                baseline=base_tile,
                blend_to_baseline=0.35,
            )
        return _sanitize_cube(current)
    raise AssertionError(f"Unhandled streaming mode: {mode}")


def _classical_tile(
    base_tile: np.ndarray,
    rgb01: np.ndarray,
    config: FusionConfig,
    mean: np.ndarray,
    basis: np.ndarray,
    *,
    strength: float | None = None,
    spatial_strength: float | None = None,
    baseline: np.ndarray | None = None,
    blend_to_baseline: float = 0.15,
) -> np.ndarray:
    h, w, bands = base_tile.shape
    coeff = _project_onto_basis(base_tile.reshape(-1, bands) - mean, basis)
    coeff = coeff.reshape(h, w, basis.shape[0]).astype(np.float32)
    coeff = _inject_rgb_detail(
        coeff,
        rgb01,
        strength=config.classical_detail_strength if strength is None else strength,
        mask=None,
        relative=False,
    )
    reconstructed = _reconstruct_low_rank(mean, basis, coeff, base_tile.shape)
    anchor = base_tile if baseline is None else baseline
    out = (1.0 - blend_to_baseline) * reconstructed + blend_to_baseline * anchor
    out = _inject_rgb_spatial_detail(
        out,
        rgb01,
        strength=config.spatial_detail_strength if spatial_strength is None else spatial_strength,
        mask=None,
        small_radius=config.spatial_detail_small_radius,
        large_radius=config.spatial_detail_large_radius,
    )
    return _sanitize_cube(out)


def _tile_margin(mode: FusionMode, config: FusionConfig) -> int:
    if mode == FusionMode.UPSAMPLE_ONLY:
        return 0
    return max(0, int(config.spatial_detail_large_radius) + 3)


def _resample_source_window(
    source: np.ndarray,
    target_shape: tuple[int, int],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    method: str,
) -> np.ndarray:
    """Resize source to a target-grid window without resizing the full source."""

    src_h, src_w, _ = source.shape
    tgt_h, tgt_w = target_shape
    yy = (np.arange(y0, y1, dtype=np.float32) + 0.5) * src_h / tgt_h - 0.5
    xx = (np.arange(x0, x1, dtype=np.float32) + 0.5) * src_w / tgt_w - 0.5
    if method.lower() == "nearest":
        yi = np.clip(np.round(yy).astype(np.int64), 0, src_h - 1)
        xi = np.clip(np.round(xx).astype(np.int64), 0, src_w - 1)
        return np.asarray(source[yi][:, xi], dtype=np.float32)

    y_floor = np.floor(yy).astype(np.int64)
    x_floor = np.floor(xx).astype(np.int64)
    y0s = int(np.clip(y_floor.min(), 0, src_h - 1))
    x0s = int(np.clip(x_floor.min(), 0, src_w - 1))
    y1s = int(np.clip((y_floor + 1).max(), 0, src_h - 1))
    x1s = int(np.clip((x_floor + 1).max(), 0, src_w - 1))
    local = np.asarray(source[y0s : y1s + 1, x0s : x1s + 1, :], dtype=np.float32)
    ly0 = np.clip(y_floor - y0s, 0, local.shape[0] - 1)
    lx0 = np.clip(x_floor - x0s, 0, local.shape[1] - 1)
    ly1 = np.clip(ly0 + 1, 0, local.shape[0] - 1)
    lx1 = np.clip(lx0 + 1, 0, local.shape[1] - 1)
    wy = (yy - y_floor).astype(np.float32)
    wx = (xx - x_floor).astype(np.float32)
    top = local[ly0][:, lx0] * (1.0 - wx)[None, :, None] + local[ly0][:, lx1] * wx[None, :, None]
    bottom = local[ly1][:, lx0] * (1.0 - wx)[None, :, None] + local[ly1][:, lx1] * wx[None, :, None]
    return top * (1.0 - wy)[:, None, None] + bottom * wy[:, None, None]


def _sample_paired_spectra(
    nir_arr: np.ndarray,
    swir_arr: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(min(max_samples, nir_arr.shape[0] * nir_arr.shape[1], swir_arr.shape[0] * swir_arr.shape[1]))
    if n <= 0:
        raise ValueError("Cannot sample empty sensor arrays")
    fy = rng.random(n)
    fx = rng.random(n)
    ny = np.clip(np.round(fy * (nir_arr.shape[0] - 1)).astype(np.int64), 0, nir_arr.shape[0] - 1)
    nx = np.clip(np.round(fx * (nir_arr.shape[1] - 1)).astype(np.int64), 0, nir_arr.shape[1] - 1)
    sy = np.clip(np.round(fy * (swir_arr.shape[0] - 1)).astype(np.int64), 0, swir_arr.shape[0] - 1)
    sx = np.clip(np.round(fx * (swir_arr.shape[1] - 1)).astype(np.int64), 0, swir_arr.shape[1] - 1)
    nir_sample = np.asarray(nir_arr[ny, nx, :], dtype=np.float32)
    swir_sample = np.asarray(swir_arr[sy, sx, :], dtype=np.float32)
    return np.nan_to_num(nir_sample), np.nan_to_num(swir_sample)


def _fit_global_basis(sample: np.ndarray, *, rank: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sample, dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    centered = np.nan_to_num(arr - mean)
    k = max(1, min(int(rank), centered.shape[1], centered.shape[0]))
    basis = _stable_pca_basis(centered, k, seed=seed)
    return mean, basis.astype(np.float32)


def _estimate_reflectance_scale(sample: np.ndarray) -> float:
    finite = np.asarray(sample)[np.isfinite(sample)]
    if finite.size == 0:
        return 1.0
    p99 = float(np.percentile(finite, 99))
    if p99 > 5.0:
        return max(p99, 1.0)
    return 1.0


def _prepare_output_dirs(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    (out / "exports").mkdir(exist_ok=True)


def _write_streaming_previews(rgb_arr: np.ndarray, preview_dir: Path, max_size: int) -> dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    h, w = rgb_arr.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max(1, max_size))))
    rgb_small = np.asarray(rgb_arr[::step, ::step, :3])
    rgb_small = (normalize_rgb(rgb_small) * 255.0).round().astype(np.uint8)
    Image.fromarray(rgb_small).save(preview_dir / "preview_rgb.png")
    return {"preview_rgb": "previews/preview_rgb.png"}


def _manifest(
    out: Path,
    mode: FusionMode,
    config: FusionConfig,
    project: ProjectMetadata | None,
    cube_shape: tuple[int, int, int],
    preview_paths: dict[str, str],
    *,
    export_envi_available: bool,
) -> dict[str, Any]:
    del out
    return {
        "schema_version": "m0_fusion_result.v1",
        "module": "M0-1",
        "algorithm": "GC-HyFuse",
        "algorithm_mode": mode.value,
        "algorithm_version": config.algorithm_version,
        "backend": "streaming",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "project": _project_to_dict(project),
        "grid": {
            "reference": "rgb",
            "height": int(cube_shape[0]),
            "width": int(cube_shape[1]),
            "axis_order": "y,x,band",
            "pixel_unit": "rgb_pixel",
        },
        "cube": {
            "path": "fused_cube.zarr",
            "array": "reflectance",
            "dtype": str(np.dtype(config.output_dtype)),
            "shape": [int(x) for x in cube_shape],
            "chunks": [int(x) for x in config.chunk_size],
            "unit": "reflectance",
            "storage": "zarr_v2_uncompressed_streaming",
        },
        "metadata": {
            "band_metadata": "metadata/band_metadata.csv",
            "registration_model": "metadata/registration_model.json",
            "spectral_stitch_model": "metadata/spectral_stitch_model.json",
            "processing_config": "metadata/processing_config.json",
            "input_metadata": "metadata/input_metadata.json",
            "quality_report": "metrics/quality_report.json",
        },
        "previews": preview_paths,
        "compat_exports": {
            "envi_dat": "exports/fused_envi.dat",
            "envi_hdr": "exports/fused_envi.hdr",
            "available": export_envi_available,
        },
    }


def _algorithm_details(mode: FusionMode, config: FusionConfig, basis: np.ndarray) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "backend": "streaming",
        "rank": int(basis.shape[0]),
        "interpolation": config.interpolation,
        "fast_detail_strength": config.fast_detail_strength,
        "classical_detail_strength": config.classical_detail_strength,
        "deep_detail_strength": config.deep_detail_strength,
        "spatial_detail_strength": config.spatial_detail_strength,
        "spatial_detail_small_radius": config.spatial_detail_small_radius,
        "spatial_detail_large_radius": config.spatial_detail_large_radius,
        "deep_iterations": config.deep_iterations,
    }


def _streaming_warnings(config: FusionConfig) -> list[str]:
    warnings = []
    if config.export_envi:
        warnings.append("Streaming backend writes Zarr output; full ENVI export is disabled to avoid a second huge write.")
    if config.chunk_size[0] >= 512 or config.chunk_size[1] >= 512:
        warnings.append("Large spatial chunks may require substantial memory; 256x256x32 is recommended for full-size jobs.")
    return warnings


def _registration_model(shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "schema_version": "registration_model.v1",
        "reference_grid": "rgb",
        "grid_shape": [int(shape[0]), int(shape[1])],
        "transforms": [
            {"name": "nir_to_rgb", "type": "streaming_scale_mapping", "direction": "source_to_reference"},
            {"name": "swir_to_rgb", "type": "streaming_scale_mapping", "direction": "source_to_reference"},
        ],
        "downstream_transform_stack": [],
    }


def _meta_to_dict(meta: Any, input_pair: EnviInput) -> dict[str, Any]:
    return {
        "hdr_path": str(input_pair.hdr_path),
        "dat_path": str(input_pair.dat_path) if input_pair.dat_path is not None else None,
        "samples": int(meta.samples),
        "lines": int(meta.lines),
        "bands": int(meta.bands),
        "data_type": int(meta.data_type),
        "interleave": meta.interleave,
        "byte_order": int(meta.byte_order),
        "header_offset": int(meta.header_offset),
        "wavelength_count": len(meta.wavelengths or []),
    }


def _config_to_dict(config: FusionConfig) -> dict[str, Any]:
    return {
        "mode": config.normalized_mode().value,
        "interpolation": config.interpolation,
        "output_dtype": config.output_dtype,
        "chunk_size": list(config.chunk_size),
        "rank": config.rank,
        "max_basis_pixels": config.max_basis_pixels,
        "fast_detail_strength": config.fast_detail_strength,
        "classical_detail_strength": config.classical_detail_strength,
        "deep_detail_strength": config.deep_detail_strength,
        "spatial_detail_strength": config.spatial_detail_strength,
        "spatial_detail_small_radius": config.spatial_detail_small_radius,
        "spatial_detail_large_radius": config.spatial_detail_large_radius,
        "deep_iterations": config.deep_iterations,
        "back_projection_iters": config.back_projection_iters,
        "back_projection_weight": config.back_projection_weight,
        "export_envi": config.export_envi,
        "write_previews": config.write_previews,
        "streaming": config.streaming,
        "streaming_preview_max_size": config.streaming_preview_max_size,
        "extra": config.extra,
    }


def _project_to_dict(project: ProjectMetadata | None) -> dict[str, Any]:
    if project is None:
        return {}
    return {
        "project_id": project.project_id,
        "borehole_id": project.borehole_id,
        "box_id": project.box_id,
        "depth_start_m": project.depth_start_m,
        "depth_end_m": project.depth_end_m,
    }


def _safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))
