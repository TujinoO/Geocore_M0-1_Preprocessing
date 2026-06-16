"""Quality metrics for fusion outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .preprocess import rgb_luminance
from .resample import downsample_to


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.nanmean(diff * diff)))


def mean_sam_deg(a: np.ndarray, b: np.ndarray, *, max_pixels: int = 20000) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1, a.shape[-1])
    bb = np.asarray(b, dtype=np.float32).reshape(-1, b.shape[-1])
    valid = np.isfinite(aa).all(axis=1) & np.isfinite(bb).all(axis=1)
    aa = aa[valid]
    bb = bb[valid]
    if aa.size == 0:
        return float("nan")
    if aa.shape[0] > max_pixels:
        idx = np.linspace(0, aa.shape[0] - 1, max_pixels).round().astype(int)
        aa = aa[idx]
        bb = bb[idx]
    dot = np.sum(aa * bb, axis=1)
    norm = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    keep = norm > 1e-12
    if keep.sum() == 0:
        return float("nan")
    cos = np.clip(dot[keep] / norm[keep], -1.0, 1.0)
    return float(np.degrees(np.nanmean(np.arccos(cos))))


def mean_band_cc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1, a.shape[-1])
    bb = np.asarray(b, dtype=np.float32).reshape(-1, b.shape[-1])
    ccs: list[float] = []
    for band in range(aa.shape[1]):
        x = aa[:, band]
        y = bb[:, band]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 4:
            continue
        x = x[valid]
        y = y[valid]
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx <= 1e-12 or sy <= 1e-12:
            continue
        ccs.append(float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)))
    return float(np.mean(ccs)) if ccs else float("nan")


def edge_correlation(cube: np.ndarray, rgb: np.ndarray) -> float:
    intensity = np.nanmean(np.asarray(cube, dtype=np.float32), axis=2)
    lum = rgb_luminance(rgb)
    gi = _grad_mag(intensity)
    gl = _grad_mag(lum)
    return _corr(gi.reshape(-1), gl.reshape(-1))


def build_quality_report(
    fused: np.ndarray,
    rgb: np.ndarray,
    nir_ref: np.ndarray,
    swir_ref: np.ndarray,
    *,
    nir_band_count: int,
    algorithm_details: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact quality report using sensor-consistency checks."""

    fused_nir = fused[:, :, :nir_band_count]
    fused_swir = fused[:, :, nir_band_count:]
    down_nir = downsample_to(fused_nir, nir_ref.shape[:2])
    down_swir = downsample_to(fused_swir, swir_ref.shape[:2])
    spectral = {
        "nir_rmse": rmse(down_nir, nir_ref),
        "swir_rmse": rmse(down_swir, swir_ref),
        "nir_sam_mean_deg": mean_sam_deg(down_nir, nir_ref),
        "swir_sam_mean_deg": mean_sam_deg(down_swir, swir_ref),
        "nir_band_cc_mean": mean_band_cc(down_nir, nir_ref),
        "swir_band_cc_mean": mean_band_cc(down_swir, swir_ref),
    }
    spatial = {
        "rgb_edge_correlation": edge_correlation(fused, rgb),
    }
    return {
        "summary": {
            "status": "passed" if _report_passed(spectral) else "warning",
            "notes": "Metrics are computed by degrading fused output back to original sensor grids.",
        },
        "spectral": spectral,
        "spatial": spatial,
        "algorithm": algorithm_details,
    }


def _grad_mag(image: np.ndarray) -> np.ndarray:
    gy = np.zeros_like(image, dtype=np.float32)
    gx = np.zeros_like(image, dtype=np.float32)
    gy[1:-1] = 0.5 * (image[2:] - image[:-2])
    gx[:, 1:-1] = 0.5 * (image[:, 2:] - image[:, :-2])
    return np.sqrt(gx * gx + gy * gy)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 4:
        return float("nan")
    x = a[valid].astype(np.float32)
    y = b[valid].astype(np.float32)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _report_passed(spectral: dict[str, float]) -> bool:
    rmse_values = [spectral.get("nir_rmse"), spectral.get("swir_rmse")]
    return all(value is not None and np.isfinite(value) and value < 0.20 for value in rmse_values)

