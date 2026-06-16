"""Command line interface for the M0-1 fusion module."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import EnviInput, FusionConfig, FusionMode, ProjectMetadata
from .pipeline import fuse_arrays, fuse_envi_files
from .resample import downsample_to


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = FusionConfig(
        mode=args.mode,
        rank=args.rank,
        chunk_size=tuple(args.chunk_size),
        export_envi=args.export_envi,
        write_previews=not args.no_previews,
        fast_detail_strength=args.fast_detail_strength,
        classical_detail_strength=args.classical_detail_strength,
        deep_detail_strength=args.deep_detail_strength,
        spatial_detail_strength=args.spatial_detail_strength,
        spatial_detail_small_radius=args.spatial_detail_small_radius,
        spatial_detail_large_radius=args.spatial_detail_large_radius,
        deep_iterations=args.deep_iterations,
        streaming=args.streaming,
        streaming_preview_max_size=args.preview_max_size,
    )
    project = ProjectMetadata(
        project_id=args.project_id,
        borehole_id=args.borehole_id,
        box_id=args.box_id,
        depth_start_m=args.depth_start,
        depth_end_m=args.depth_end,
    )
    output = Path(args.output)
    if args.demo:
        rgb, nir, swir, nir_w, swir_w = make_demo_inputs()
        result = fuse_arrays(
            rgb,
            nir,
            swir,
            output_dir=output,
            nir_wavelengths=nir_w,
            swir_wavelengths=swir_w,
            config=config,
            project=project,
            input_metadata={"source": "synthetic_demo"},
        )
    else:
        required = [
            args.rgb_hdr,
            args.nir_hdr,
            args.swir_hdr,
        ]
        if any(item is None for item in required):
            parser.error("Real-data mode requires --rgb-hdr, --nir-hdr, and --swir-hdr")
        result = fuse_envi_files(
            EnviInput(Path(args.rgb_hdr), Path(args.rgb_dat) if args.rgb_dat else None),
            EnviInput(Path(args.nir_hdr), Path(args.nir_dat) if args.nir_dat else None),
            EnviInput(Path(args.swir_hdr), Path(args.swir_dat) if args.swir_dat else None),
            output_dir=output,
            config=config,
            project=project,
            mmap=not args.no_mmap,
        )
    print(f"M0-1 fusion completed: {result.output_dir / 'manifest.json'}")
    print(f"mode={result.manifest['algorithm_mode']} shape={result.manifest['cube']['shape']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Geo-Core AI M0-1 hyperspectral-optical fusion")
    parser.add_argument("--mode", choices=[m.value for m in FusionMode], default=FusionMode.CLASSICAL.value)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic demo inputs")
    parser.add_argument("--rgb-hdr")
    parser.add_argument("--rgb-dat")
    parser.add_argument("--nir-hdr")
    parser.add_argument("--nir-dat")
    parser.add_argument("--swir-hdr")
    parser.add_argument("--swir-dat")
    parser.add_argument("--no-mmap", action="store_true")
    parser.add_argument("--export-envi", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Use out-of-core block streaming for ENVI inputs")
    parser.add_argument("--preview-max-size", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=12)
    parser.add_argument("--deep-iterations", type=int, default=4)
    parser.add_argument("--fast-detail-strength", type=float, default=0.26)
    parser.add_argument("--classical-detail-strength", type=float, default=0.12)
    parser.add_argument("--deep-detail-strength", type=float, default=0.10)
    parser.add_argument("--spatial-detail-strength", type=float, default=0.42)
    parser.add_argument("--spatial-detail-small-radius", type=int, default=2)
    parser.add_argument("--spatial-detail-large-radius", type=int, default=9)
    parser.add_argument("--chunk-size", type=int, nargs=3, default=[512, 512, 32])
    parser.add_argument("--project-id")
    parser.add_argument("--borehole-id")
    parser.add_argument("--box-id")
    parser.add_argument("--depth-start", type=float)
    parser.add_argument("--depth-end", type=float)
    return parser


def make_demo_inputs(
    *,
    height: int = 96,
    width: int = 128,
    nir_shape: tuple[int, int] = (24, 32),
    swir_shape: tuple[int, int] = (25, 32),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    """Create deterministic synthetic data with RGB texture and spectral structure."""

    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    layer = (np.sin(2 * np.pi * (3 * y + 0.4 * x)) + 1.0) * 0.5
    vein = np.exp(-((x - 0.62) ** 2) / 0.002) * (0.35 + 0.65 * y)
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * (17 * x + 11 * y))
    coeffs = np.stack([layer, vein, grain], axis=2).astype(np.float32)
    wavelengths = np.array([700, 760, 850, 970, 1100, 1250, 1400, 1600, 1900, 2200, 2350, 2500], dtype=np.float32)
    basis = np.stack(
        [
            0.25 + 0.25 * np.exp(-((wavelengths - 850) / 260) ** 2),
            0.12 + 0.55 * np.exp(-((wavelengths - 2200) / 180) ** 2),
            0.18 + 0.18 * np.sin(wavelengths / 180),
        ],
        axis=0,
    ).astype(np.float32)
    cube = (
        coeffs[:, :, 0:1] * basis[0][None, None, :]
        + coeffs[:, :, 1:2] * basis[1][None, None, :]
        + coeffs[:, :, 2:3] * basis[2][None, None, :]
    ).astype(np.float32)
    cube = np.clip(cube, 0.0, 1.0)
    rgb = np.stack(
        [
            np.clip(0.25 + 0.7 * layer + 0.2 * grain, 0, 1),
            np.clip(0.20 + 0.45 * layer + 0.4 * vein, 0, 1),
            np.clip(0.15 + 0.25 * grain + 0.1 * vein, 0, 1),
        ],
        axis=2,
    )
    rgb = (rgb * 255).round().astype(np.uint8)
    nir = downsample_to(cube[:, :, :7], nir_shape)
    swir = downsample_to(cube[:, :, 5:], swir_shape)
    nir_w = wavelengths[:7].astype(float).tolist()
    swir_w = wavelengths[5:].astype(float).tolist()
    return rgb, nir.astype(np.float32), swir.astype(np.float32), nir_w, swir_w


if __name__ == "__main__":
    raise SystemExit(main())
