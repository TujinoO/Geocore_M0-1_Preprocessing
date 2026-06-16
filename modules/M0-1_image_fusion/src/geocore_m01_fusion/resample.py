"""Small NumPy-only resampling helpers."""

from __future__ import annotations

import numpy as np


def resize_cube(
    cube: np.ndarray,
    target_shape: tuple[int, int],
    *,
    method: str = "bilinear",
) -> np.ndarray:
    """Resize a 2-D image or y,x,band cube to target height and width."""

    arr = np.asarray(cube)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError("resize_cube expects a 2-D image or y,x,band cube")

    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    h, w, _ = arr.shape
    if (h, w) == (target_h, target_w):
        out = arr.copy()
        return out[:, :, 0] if squeeze else out

    method = method.lower()
    if method == "nearest":
        out = _resize_nearest(arr, target_h, target_w)
    elif method in {"bilinear", "linear"}:
        out = _resize_bilinear(arr, target_h, target_w)
    else:
        raise ValueError(f"Unsupported resize method: {method}")
    return out[:, :, 0] if squeeze else out


def downsample_to(cube: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Downsample using area averaging when possible, otherwise bilinear resize."""

    arr = np.asarray(cube)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze = True
    h, w, _ = arr.shape
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_shape must be positive")
    if h % target_h == 0 and w % target_w == 0:
        fh = h // target_h
        fw = w // target_w
        out = arr.reshape(target_h, fh, target_w, fw, arr.shape[2]).mean(axis=(1, 3))
    else:
        out = resize_cube(arr, (target_h, target_w), method="bilinear")
    return out[:, :, 0] if squeeze else out


def box_blur2d(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Reflect-padded box blur for a 2-D image."""

    img = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return img.copy()
    pad = int(radius)
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    k = 2 * pad + 1
    total = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    return total / float(k * k)


def _resize_nearest(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w, _ = arr.shape
    y = np.clip(np.round((np.arange(target_h) + 0.5) * h / target_h - 0.5), 0, h - 1)
    x = np.clip(np.round((np.arange(target_w) + 0.5) * w / target_w - 0.5), 0, w - 1)
    return arr[y.astype(np.int64)][:, x.astype(np.int64)]


def _resize_bilinear(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w, _ = arr.shape
    y = (np.arange(target_h, dtype=np.float32) + 0.5) * h / target_h - 0.5
    x = (np.arange(target_w, dtype=np.float32) + 0.5) * w / target_w - 0.5
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)

    top = arr[y0][:, x0] * (1.0 - wx)[None, :, None] + arr[y0][:, x1] * wx[None, :, None]
    bottom = arr[y1][:, x0] * (1.0 - wx)[None, :, None] + arr[y1][:, x1] * wx[None, :, None]
    return top * (1.0 - wy)[:, None, None] + bottom * wy[:, None, None]

