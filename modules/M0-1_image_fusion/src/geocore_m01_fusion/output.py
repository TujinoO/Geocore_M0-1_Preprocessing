"""Output writers for M0-1 fusion products."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .algorithms import FusionProduct
from .config import FusionConfig, ProjectMetadata
from .envi import write_envi
from .preprocess import normalize_rgb


def write_fusion_output(
    product: FusionProduct,
    rgb: np.ndarray,
    output_dir: str | Path,
    *,
    config: FusionConfig,
    project: ProjectMetadata | None,
    quality_report: dict[str, Any],
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the full output contract and return the manifest."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)
    (out / "previews").mkdir(exist_ok=True)
    (out / "exports").mkdir(exist_ok=True)

    zarr_path = out / "fused_cube.zarr"
    write_zarr_v2_uncompressed(
        product.cube.astype(config.output_dtype, copy=False),
        zarr_path,
        chunks=config.chunk_size,
        attrs={
            "array_name": "reflectance",
            "axis_order": "y,x,band",
            "unit": "reflectance",
            "algorithm_mode": product.algorithm_details.get("mode"),
        },
    )

    band_metadata_path = out / "metadata" / "band_metadata.csv"
    write_band_metadata(product.band_metadata, band_metadata_path)

    stitch_path = out / "metadata" / "spectral_stitch_model.json"
    write_json(product.stitch_model, stitch_path)

    input_metadata_path = out / "metadata" / "input_metadata.json"
    write_json(input_metadata or {}, input_metadata_path)

    processing_config_path = out / "metadata" / "processing_config.json"
    write_json(_config_to_dict(config), processing_config_path)

    registration_model = _identity_registration_model(product.cube.shape[:2])
    registration_path = out / "metadata" / "registration_model.json"
    write_json(registration_model, registration_path)

    quality_path = out / "metrics" / "quality_report.json"
    write_json(quality_report, quality_path)

    preview_paths = {}
    if config.write_previews:
        preview_paths = write_previews(product.cube, rgb, out / "previews")

    envi_available = False
    envi_dat = out / "exports" / "fused_envi.dat"
    envi_hdr = out / "exports" / "fused_envi.hdr"
    if config.export_envi:
        write_envi(product.cube, envi_hdr, envi_dat, wavelengths=product.wavelengths)
        envi_available = True

    manifest = {
        "schema_version": "m0_fusion_result.v1",
        "module": "M0-1",
        "algorithm": "GC-HyFuse",
        "algorithm_mode": product.algorithm_details.get("mode"),
        "algorithm_version": config.algorithm_version,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "project": _project_to_dict(project),
        "grid": {
            "reference": "rgb",
            "height": int(product.cube.shape[0]),
            "width": int(product.cube.shape[1]),
            "axis_order": "y,x,band",
            "pixel_unit": "rgb_pixel",
        },
        "cube": {
            "path": "fused_cube.zarr",
            "array": "reflectance",
            "dtype": str(product.cube.astype(config.output_dtype, copy=False).dtype),
            "shape": [int(x) for x in product.cube.shape],
            "chunks": [int(x) for x in config.chunk_size],
            "unit": "reflectance",
            "storage": "zarr_v2_uncompressed",
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
            "available": envi_available,
        },
    }
    write_json(manifest, out / "manifest.json")
    return manifest


def write_zarr_v2_uncompressed(
    array: np.ndarray,
    path: str | Path,
    *,
    chunks: tuple[int, int, int],
    attrs: dict[str, Any] | None = None,
    fill_value: float = 0.0,
) -> None:
    """Write a minimal uncompressed Zarr v2 array.

    This avoids requiring the zarr package while keeping the on-disk layout
    readable by standard Zarr implementations.
    """

    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError("Zarr writer expects a y,x,band array")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    chunks = tuple(int(c) for c in chunks)
    metadata = {
        "zarr_format": 2,
        "shape": [int(x) for x in arr.shape],
        "chunks": [int(x) for x in chunks],
        "dtype": np.dtype(arr.dtype).str,
        "compressor": None,
        "fill_value": fill_value,
        "order": "C",
        "filters": None,
    }
    write_json(metadata, root / ".zarray")
    write_json(attrs or {}, root / ".zattrs")

    h, w, b = arr.shape
    cy, cx, cb = chunks
    for y0 in range(0, h, cy):
        for x0 in range(0, w, cx):
            for b0 in range(0, b, cb):
                chunk = np.full(chunks, fill_value, dtype=arr.dtype)
                part = arr[y0 : y0 + cy, x0 : x0 + cx, b0 : b0 + cb]
                chunk[: part.shape[0], : part.shape[1], : part.shape[2]] = part
                chunk_path = root / f"{y0 // cy}.{x0 // cx}.{b0 // cb}"
                np.ascontiguousarray(chunk).tofile(chunk_path)


class ZarrV2ChunkWriter:
    """Incremental writer for spatial tiles aligned with Zarr chunks."""

    def __init__(
        self,
        path: str | Path,
        *,
        shape: tuple[int, int, int],
        chunks: tuple[int, int, int],
        dtype: str | np.dtype = "float32",
        attrs: dict[str, Any] | None = None,
        fill_value: float = 0.0,
    ) -> None:
        self.root = Path(path)
        self.shape = tuple(int(x) for x in shape)
        self.chunks = tuple(int(x) for x in chunks)
        self.dtype = np.dtype(dtype)
        self.fill_value = fill_value
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "zarr_format": 2,
            "shape": [int(x) for x in self.shape],
            "chunks": [int(x) for x in self.chunks],
            "dtype": self.dtype.str,
            "compressor": None,
            "fill_value": fill_value,
            "order": "C",
            "filters": None,
        }
        write_json(metadata, self.root / ".zarray")
        write_json(attrs or {}, self.root / ".zattrs")

    def write_tile(self, y0: int, x0: int, tile: np.ndarray) -> None:
        """Write a tile whose spatial origin is chunk-aligned."""

        cy, cx, cb = self.chunks
        if y0 % cy != 0 or x0 % cx != 0:
            raise ValueError("Streaming tile origins must align with spatial chunk size")
        arr = np.asarray(tile, dtype=self.dtype)
        if arr.ndim != 3:
            raise ValueError("Streaming tile must be y,x,band")
        y1 = min(y0 + arr.shape[0], self.shape[0])
        x1 = min(x0 + arr.shape[1], self.shape[1])
        if arr.shape[0] != y1 - y0 or arr.shape[1] != x1 - x0 or arr.shape[2] != self.shape[2]:
            raise ValueError("Streaming tile shape does not match destination array window")
        for b0 in range(0, self.shape[2], cb):
            b1 = min(b0 + cb, self.shape[2])
            chunk = np.full(self.chunks, self.fill_value, dtype=self.dtype)
            part = arr[:, :, b0:b1]
            chunk[: part.shape[0], : part.shape[1], : part.shape[2]] = part
            chunk_path = self.root / f"{y0 // cy}.{x0 // cx}.{b0 // cb}"
            np.ascontiguousarray(chunk).tofile(chunk_path)


def read_zarr_v2_uncompressed(path: str | Path) -> np.ndarray:
    """Read arrays produced by write_zarr_v2_uncompressed."""

    root = Path(path)
    meta = json.loads((root / ".zarray").read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in meta["shape"])
    chunks = tuple(int(x) for x in meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    fill = meta.get("fill_value", 0.0)
    out = np.full(shape, fill, dtype=dtype)
    h, w, b = shape
    cy, cx, cb = chunks
    for y0 in range(0, h, cy):
        for x0 in range(0, w, cx):
            for b0 in range(0, b, cb):
                chunk_path = root / f"{y0 // cy}.{x0 // cx}.{b0 // cb}"
                chunk = np.fromfile(chunk_path, dtype=dtype).reshape(chunks)
                y1, x1, b1 = min(y0 + cy, h), min(x0 + cx, w), min(b0 + cb, b)
                out[y0:y1, x0:x1, b0:b1] = chunk[: y1 - y0, : x1 - x0, : b1 - b0]
    return out


def write_band_metadata(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("band metadata cannot be empty")
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_previews(cube: np.ndarray, rgb: np.ndarray, preview_dir: Path) -> dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    rgb_img = (normalize_rgb(rgb) * 255.0).round().astype(np.uint8)
    _save_png(rgb_img, preview_dir / "preview_rgb.png")
    paths["preview_rgb"] = "previews/preview_rgb.png"

    if cube.shape[2] >= 3:
        false_color = _false_color(cube, [0, cube.shape[2] // 2, cube.shape[2] - 1])
        _save_png(false_color, preview_dir / "preview_falsecolor_full.png")
        paths["preview_falsecolor_full"] = "previews/preview_falsecolor_full.png"

    intensity = _stretch(np.nanmean(cube, axis=2))
    _save_png(intensity, preview_dir / "preview_mean_reflectance.png")
    paths["preview_mean_reflectance"] = "previews/preview_mean_reflectance.png"
    return paths


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _false_color(cube: np.ndarray, bands: list[int]) -> np.ndarray:
    channels = [_stretch(cube[:, :, int(np.clip(b, 0, cube.shape[2] - 1))]) for b in bands]
    return np.stack(channels, axis=2)


def _stretch(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).round().astype(np.uint8)


def _save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path)


def _identity_registration_model(shape: tuple[int, int]) -> dict[str, Any]:
    return {
        "schema_version": "registration_model.v1",
        "reference_grid": "rgb",
        "grid_shape": [int(shape[0]), int(shape[1])],
        "transforms": [
            {
                "name": "nir_to_rgb",
                "type": "identity_or_prealigned",
                "direction": "source_to_reference",
                "rmse_pixels_rgb": None,
            },
            {
                "name": "swir_to_rgb",
                "type": "identity_or_prealigned",
                "direction": "source_to_reference",
                "rmse_pixels_rgb": None,
            },
        ],
        "downstream_transform_stack": [],
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
