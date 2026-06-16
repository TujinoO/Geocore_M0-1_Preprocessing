"""Preprocessing helpers for reflectance and RGB guidance."""

from __future__ import annotations

import numpy as np

from .resample import box_blur2d


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return RGB as float32 in 0-1 range."""

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("RGB input must have shape y,x,3")
    arr = arr[:, :, :3].astype(np.float32, copy=False)
    if np.issubdtype(np.asarray(rgb).dtype, np.integer):
        max_value = np.iinfo(np.asarray(rgb).dtype).max
        arr = arr / float(max_value)
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size and finite.max() > 2.0:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def normalize_reflectance(cube: np.ndarray) -> np.ndarray:
    """Return a float32 reflectance-like cube with finite values."""

    arr = np.asarray(cube).astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = arr.copy()
        arr[~finite] = np.nan
    if np.nanmax(arr) > 5.0:
        p99 = np.nanpercentile(arr, 99)
        if p99 > 0:
            arr = arr / float(p99)
    return arr


def rgb_luminance(rgb01: np.ndarray) -> np.ndarray:
    """Compute RGB luminance from normalized RGB."""

    rgb = normalize_rgb(rgb01)
    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


def edge_weight_from_rgb(rgb01: np.ndarray) -> np.ndarray:
    """Compute normalized RGB edge weights in 0-1 range."""

    lum = rgb_luminance(rgb01)
    gy = np.zeros_like(lum)
    gx = np.zeros_like(lum)
    gy[1:-1] = 0.5 * (lum[2:] - lum[:-2])
    gx[:, 1:-1] = 0.5 * (lum[:, 2:] - lum[:, :-2])
    grad = np.sqrt(gx * gx + gy * gy)
    scale = np.percentile(grad, 99)
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(grad)
    return np.clip(grad / scale, 0.0, 1.0).astype(np.float32)


def rgb_detail(rgb01: np.ndarray, blur_radius: int = 5) -> np.ndarray:
    """Extract normalized high-frequency detail from RGB luminance."""

    lum = rgb_luminance(rgb01)
    detail = lum - box_blur2d(lum, radius=blur_radius)
    std = float(np.std(detail))
    if std <= 1e-6:
        return np.zeros_like(detail, dtype=np.float32)
    return np.clip(detail / (3.0 * std), -1.0, 1.0).astype(np.float32)


def normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Return a float mask matching target spatial shape."""

    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape != shape:
        from .resample import resize_cube

        arr = resize_cube(arr, shape, method="nearest")
    return (arr > 0.5).astype(np.float32)

