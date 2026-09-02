from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT = Path(r"E:\Code\GeoCoreFusion")
MODULE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from geocorefusion.config import load_config  # noqa: E402
from geocorefusion.dataset import discover_triplet, normalize_image, rgb_structure  # noqa: E402
from geocorefusion.registration import (  # noqa: E402
    analysis_rgb_grid,
    estimate_registration,
    sample_cube_on_rgb_grid,
)


def edge(image: np.ndarray) -> np.ndarray:
    image = normalize_image(image)
    gx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    return normalize_image(cv2.magnitude(gx, gy))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 0 else float("nan")


def save_overlay(path: Path, reference: np.ndarray, moving: np.ndarray) -> None:
    ref = normalize_image(reference)
    mov = normalize_image(moving)
    overlay = np.stack([ref, mov, 0.5 * (ref + mov)], axis=2)
    Image.fromarray((np.clip(overlay, 0, 1) * 255).round().astype(np.uint8)).save(path)


def run(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    dataset = discover_triplet(config.data_dir)
    registration = estimate_registration(dataset, config.registration)
    roi = {
        "x": int(config.roi.x or 0),
        "y": int(config.roi.y or 0),
        "width": int(config.roi.width),
        "height": int(config.roi.height),
    }
    manifest_path = config.output_dir / "manifest.json"
    if manifest_path.exists():
        roi = json.loads(manifest_path.read_text(encoding="utf-8"))["rgb_roi"]

    output.mkdir(parents=True, exist_ok=True)
    out_shape = (roi["height"], roi["width"])
    yy, xx = analysis_rgb_grid(roi, *out_shape)
    rgb_crop = np.asarray(
        dataset.rgb.cube[
            roi["y"] : roi["y"] + roi["height"],
            roi["x"] : roi["x"] + roi["width"],
            :3,
        ]
    )
    rgb_img = rgb_structure(rgb_crop, target_shape=out_shape)

    report: dict[str, dict[str, float]] = {}
    for name, sensor, model, wavelengths in (
        ("nir", dataset.nir, registration.nir, [850, 1050, 1250, 1400]),
        ("swir", dataset.swir, registration.swir, [1050, 1250, 1650, 2200]),
    ):
        band_ids = [int(np.argmin(np.abs(sensor.meta.wavelengths - w))) for w in wavelengths]
        aligned = sample_cube_on_rgb_grid(sensor.cube, model, yy, xx, bands=band_ids)
        image = normalize_image(np.nanmean(aligned, axis=2))
        save_overlay(output / f"{name}_roi_with_drift.png", edge(rgb_img), edge(image))
        with_dx, with_dy = model.drift_sensor_dx.copy(), model.drift_sensor_dy.copy()
        model.drift_sensor_dx = np.zeros(0, dtype=np.float32)
        model.drift_sensor_dy = np.zeros(0, dtype=np.float32)
        model.drift_rgb_y = np.zeros(0, dtype=np.float32)
        aligned_no = sample_cube_on_rgb_grid(sensor.cube, model, yy, xx, bands=band_ids)
        image_no = normalize_image(np.nanmean(aligned_no, axis=2))
        save_overlay(output / f"{name}_roi_affine_only.png", edge(rgb_img), edge(image_no))
        report[name] = {
            "edge_corr_with_drift": corr(edge(rgb_img), edge(image)),
            "edge_corr_affine_only": corr(edge(rgb_img), edge(image_no)),
            "max_abs_dx_sensor": float(np.max(np.abs(with_dx))) if with_dx.size else 0.0,
            "max_abs_dy_sensor": float(np.max(np.abs(with_dy))) if with_dy.size else 0.0,
        }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = MODULE_ROOT / "runs" / "legacy_registration" / "registration_diagnostics"
    run(PROJECT / "configs" / "3dssz_roi.yaml", root / "3dssz")
    run(PROJECT / "configs" / "zkh3_roi.yaml", root / "zkh3")
