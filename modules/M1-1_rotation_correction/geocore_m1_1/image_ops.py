from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.float32)
    return (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )


def make_thumbnail(image: np.ndarray, width: int) -> tuple[np.ndarray, float, float]:
    """Resize an RGB-like image to a fixed width and return x/y scale factors."""

    arr = np.asarray(image)
    if arr.ndim == 2:
        pil = Image.fromarray(_as_uint8(arr), mode="L")
    else:
        pil = Image.fromarray(_as_uint8(arr[:, :, :3]), mode="RGB")
    src_w, src_h = pil.size
    height = max(1, int(round(src_h * width / src_w)))
    thumb = np.asarray(pil.resize((width, height), Image.Resampling.BILINEAR))
    if thumb.ndim == 2:
        thumb = thumb[:, :, None]
    return thumb, src_w / width, src_h / height


def contrast_stretch_uint8(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(image)
    lo, hi = np.percentile(arr, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def pad_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, int(x0) - pad),
        max(0, int(y0) - pad),
        min(width - 1, int(x1) + pad),
        min(height - 1, int(y1) + pad),
    )


def rotate_nearest(
    image: np.ndarray,
    angle_deg: float,
    fill_value: int | float = 0,
    row_chunk: int = 768,
) -> tuple[np.ndarray, list[list[float]]]:
    """Rotate an image visually counterclockwise by angle_deg using nearest neighbor."""

    arr = np.asarray(image)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or HWC image, got shape {arr.shape}")

    h, w, c = arr.shape
    if h == 0 or w == 0:
        raise ValueError("Cannot rotate an empty image.")

    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    corners = np.array(
        [
            [-cx, -cy],
            [w - 1 - cx, -cy],
            [-cx, h - 1 - cy],
            [w - 1 - cx, h - 1 - cy],
        ],
        dtype=np.float64,
    )
    rot = np.empty_like(corners)
    rot[:, 0] = cos_a * corners[:, 0] + sin_a * corners[:, 1]
    rot[:, 1] = -sin_a * corners[:, 0] + cos_a * corners[:, 1]
    min_x = math.floor(float(rot[:, 0].min()))
    min_y = math.floor(float(rot[:, 1].min()))
    max_x = math.ceil(float(rot[:, 0].max()))
    max_y = math.ceil(float(rot[:, 1].max()))
    out_w = int(max_x - min_x + 1)
    out_h = int(max_y - min_y + 1)

    out = np.full((out_h, out_w, c), fill_value, dtype=arr.dtype)
    x_rel = np.arange(out_w, dtype=np.float64) + min_x

    for y_start in range(0, out_h, row_chunk):
        y_end = min(out_h, y_start + row_chunk)
        y_rel = np.arange(y_start, y_end, dtype=np.float64) + min_y
        xr, yr = np.meshgrid(x_rel, y_rel)
        src_x = cos_a * xr - sin_a * yr + cx
        src_y = sin_a * xr + cos_a * yr + cy
        src_xi = np.rint(src_x).astype(np.int64)
        src_yi = np.rint(src_y).astype(np.int64)
        valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)
        if np.any(valid):
            out_block = out[y_start:y_end]
            out_block[valid] = arr[src_yi[valid], src_xi[valid]]

    matrix = [
        [
            float(cos_a),
            float(sin_a),
            float(-cos_a * cx - sin_a * cy - min_x),
        ],
        [
            float(-sin_a),
            float(cos_a),
            float(sin_a * cx - cos_a * cy - min_y),
        ],
    ]
    if squeeze:
        return out[:, :, 0], matrix
    return out, matrix


def draw_detection_overlay(
    thumbnail: np.ndarray,
    boxes_xyxy: list[tuple[int, int, int, int]],
    output_path: str | Path,
    quality: int = 92,
) -> None:
    vis = contrast_stretch_uint8(thumbnail)
    if vis.ndim == 2:
        vis = np.stack([vis, vis, vis], axis=-1)
    if vis.ndim == 3 and vis.shape[2] == 1:
        vis = np.repeat(vis, 3, axis=2)
    image = Image.fromarray(vis[:, :, :3], mode="RGB")
    draw = ImageDraw.Draw(image)
    for idx, (x0, y0, x1, y1) in enumerate(boxes_xyxy, start=1):
        draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 30), width=3)
        draw.text((x0 + 4, y0 + 4), str(idx), fill=(255, 240, 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=quality)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return contrast_stretch_uint8(arr)

