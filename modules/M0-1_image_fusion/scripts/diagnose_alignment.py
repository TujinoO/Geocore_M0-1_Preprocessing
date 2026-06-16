from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from geocore_m01_fusion.envi import read_envi, parse_envi_header
from geocore_m01_fusion.preprocess import normalize_rgb
from geocore_m01_fusion.resample import resize_cube
from geocore_m01_fusion.roi import _hsi_structure, _select_bands, _normalize_image


def save_gray(img: np.ndarray, path: Path) -> None:
    Image.fromarray((_normalize_image(img) * 255).round().astype(np.uint8)).save(path)


def overlay(rgb_gray: np.ndarray, sensor_gray: np.ndarray, path: Path) -> None:
    rgb_norm = _normalize_image(rgb_gray)
    sensor_norm = _normalize_image(sensor_gray)
    out = np.stack([rgb_norm, sensor_norm, 0.5 * (rgb_norm + sensor_norm)], axis=2)
    Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8)).save(path)


def main() -> int:
    root = Path(os.environ["DATA_ROOT"])
    out = Path(os.environ.get("DIAG_OUT", "roi_outputs/alignment_diagnosis"))
    out.mkdir(parents=True, exist_ok=True)
    rgb_hdr = next(root.glob("RGB-*.hdr"))
    nir_hdr = next(root.glob("NIR-*.hdr"))
    swir_hdr = next(root.glob("SWIR-*.hdr"))
    rgb, _ = read_envi(rgb_hdr, mmap=True)
    nir, nir_meta = read_envi(nir_hdr, mmap=True)
    swir, swir_meta = read_envi(swir_hdr, mmap=True)
    rgb_full_gray = np.mean(normalize_rgb(np.asarray(rgb[:, :, :3])), axis=2)
    rgb_sample = rgb_full_gray[::8, ::8]
    rgb_gray = rgb_sample
    nir_struct = _hsi_structure(nir, _select_bands(nir_meta, [750, 900, 1100, 1300]))
    swir_struct = _hsi_structure(swir, _select_bands(swir_meta, [1200, 1600, 2200, 2350]))

    save_gray(rgb_gray, out / "rgb_downsample_8x.png")
    save_gray(nir_struct, out / "nir_native_struct.png")
    save_gray(swir_struct, out / "swir_native_struct.png")

    # Hypothesis A: HSI is lower-resolution full-scene coverage.
    rgb_to_nir = resize_cube(rgb_full_gray, nir_struct.shape, method="bilinear")
    rgb_to_swir = resize_cube(rgb_full_gray, swir_struct.shape, method="bilinear")
    overlay(rgb_to_nir, nir_struct, out / "overlay_full_scene_rgb_nir.png")
    overlay(rgb_to_swir, swir_struct, out / "overlay_full_scene_rgb_swir.png")

    # Hypothesis B: HSI is a native-pixel top-left subset of RGB.
    top_h, top_w = nir_struct.shape
    rgb_top_nir = rgb_full_gray[:top_h, :top_w]
    overlay(rgb_top_nir, nir_struct, out / "overlay_top_left_native_rgb_nir.png")
    top_h, top_w = swir_struct.shape
    rgb_top_swir = rgb_full_gray[:top_h, :top_w]
    overlay(rgb_top_swir, swir_struct, out / "overlay_top_left_native_rgb_swir.png")

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
