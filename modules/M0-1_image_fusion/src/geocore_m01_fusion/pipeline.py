"""Public pipeline entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import FusionProduct, fuse_cubes
from .config import EnviInput, FusionConfig, ProjectMetadata
from .envi import read_envi
from .metrics import build_quality_report
from .output import write_fusion_output


@dataclass(slots=True)
class FusionRunResult:
    product: FusionProduct | None
    quality_report: dict[str, Any]
    manifest: dict[str, Any]
    output_dir: Path


def fuse_arrays(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    *,
    output_dir: str | Path,
    nir_wavelengths: list[float] | None = None,
    swir_wavelengths: list[float] | None = None,
    config: FusionConfig | None = None,
    mask: np.ndarray | None = None,
    project: ProjectMetadata | None = None,
    input_metadata: dict[str, Any] | None = None,
) -> FusionRunResult:
    """Fuse in-memory arrays and write the standard output directory."""

    cfg = config or FusionConfig()
    product = fuse_cubes(
        rgb,
        nir,
        swir,
        nir_wavelengths=nir_wavelengths,
        swir_wavelengths=swir_wavelengths,
        config=cfg,
        mask=mask,
    )
    quality_report = build_quality_report(
        product.cube,
        rgb,
        product.nir_reference,
        product.swir_reference,
        nir_band_count=product.nir_band_count,
        algorithm_details=product.algorithm_details,
    )
    manifest = write_fusion_output(
        product,
        rgb,
        output_dir,
        config=cfg,
        project=project,
        quality_report=quality_report,
        input_metadata=input_metadata,
    )
    return FusionRunResult(
        product=product,
        quality_report=quality_report,
        manifest=manifest,
        output_dir=Path(output_dir),
    )


def fuse_envi_files(
    rgb: EnviInput,
    nir: EnviInput,
    swir: EnviInput,
    *,
    output_dir: str | Path,
    config: FusionConfig | None = None,
    mask: np.ndarray | None = None,
    project: ProjectMetadata | None = None,
    mmap: bool = True,
) -> FusionRunResult:
    """Fuse ENVI .hdr/.dat inputs."""

    cfg = config or FusionConfig()
    if cfg.streaming:
        from .streaming import fuse_envi_files_streaming

        return fuse_envi_files_streaming(
            rgb,
            nir,
            swir,
            output_dir=output_dir,
            config=cfg,
            project=project,
            mmap=mmap,
        )
    rgb_arr, rgb_meta = read_envi(rgb.hdr_path, rgb.dat_path, mmap=mmap)
    nir_arr, nir_meta = read_envi(nir.hdr_path, nir.dat_path, mmap=mmap)
    swir_arr, swir_meta = read_envi(swir.hdr_path, swir.dat_path, mmap=mmap)
    input_metadata = {
        "rgb": _meta_to_dict(rgb_meta, rgb),
        "nir": _meta_to_dict(nir_meta, nir),
        "swir": _meta_to_dict(swir_meta, swir),
    }
    return fuse_arrays(
        np.asarray(rgb_arr),
        np.asarray(nir_arr),
        np.asarray(swir_arr),
        output_dir=output_dir,
        nir_wavelengths=nir_meta.wavelengths,
        swir_wavelengths=swir_meta.wavelengths,
        config=cfg,
        mask=mask,
        project=project,
        input_metadata=input_metadata,
    )


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
