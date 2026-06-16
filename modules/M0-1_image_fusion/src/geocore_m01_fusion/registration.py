"""ENVI-style automatic registration with tie points and local warp.

The implementation mirrors the practical ENVI workflow at a small ROI scale:

1. Build modality-stable structure images.
2. Generate grid tie points.
3. Score candidate matches with edge correlation plus normalized mutual
   information, which is more robust for NIR/SWIR/RGB cross-modality data.
4. Reject outlier tie points with a median/MAD displacement filter.
5. Interpolate a dense local displacement field.
6. Warp the moving cube onto the reference grid.

The code is NumPy-only so the module remains runnable in the current bundled
environment without OpenCV/SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class TiePoint:
    ref_y: float
    ref_x: float
    moving_y: float
    moving_x: float
    approx_y: float
    approx_x: float
    score: float

    @property
    def delta_y(self) -> float:
        return self.moving_y - self.approx_y

    @property
    def delta_x(self) -> float:
        return self.moving_x - self.approx_x

    def to_dict(self) -> dict[str, float]:
        return {
            "ref_y": float(self.ref_y),
            "ref_x": float(self.ref_x),
            "moving_y": float(self.moving_y),
            "moving_x": float(self.moving_x),
            "approx_y": float(self.approx_y),
            "approx_x": float(self.approx_x),
            "delta_y": float(self.delta_y),
            "delta_x": float(self.delta_x),
            "score": float(self.score),
        }


@dataclass(slots=True)
class LocalWarpModel:
    reference_shape: tuple[int, int]
    moving_shape: tuple[int, int]
    approx_offset_y: float
    approx_offset_x: float
    approx_scale_y: float
    approx_scale_x: float
    tie_points: list[TiePoint]
    rejected_tie_points: list[TiePoint]
    method: str = "grid_tie_points_idw_warp"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "reference_shape": [int(x) for x in self.reference_shape],
            "moving_shape": [int(x) for x in self.moving_shape],
            "approx_offset_y": float(self.approx_offset_y),
            "approx_offset_x": float(self.approx_offset_x),
            "approx_scale_y": float(self.approx_scale_y),
            "approx_scale_x": float(self.approx_scale_x),
            "tie_point_count": len(self.tie_points),
            "rejected_tie_point_count": len(self.rejected_tie_points),
            "median_delta_y": _safe_median([p.delta_y for p in self.tie_points]),
            "median_delta_x": _safe_median([p.delta_x for p in self.tie_points]),
            "mean_score": _safe_mean([p.score for p in self.tie_points]),
            "tie_points": [p.to_dict() for p in self.tie_points],
            "rejected_tie_points": [p.to_dict() for p in self.rejected_tie_points[:32]],
        }


def register_local_warp(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    approx_offset_y: float = 0.0,
    approx_offset_x: float = 0.0,
    approx_scale_y: float | None = None,
    approx_scale_x: float | None = None,
    grid_rows: int = 7,
    grid_cols: int = 5,
    template_radius: int = 8,
    search_radius_y: int = 24,
    search_radius_x: int = 12,
    min_tie_points: int = 8,
) -> LocalWarpModel:
    """Register a moving structure image to a reference structure image."""

    ref = _normalize_image(reference)
    mov = _normalize_image(moving)
    if ref.ndim != 2 or mov.ndim != 2:
        raise ValueError("register_local_warp expects 2-D structure images")
    ref_h, ref_w = ref.shape
    mov_h, mov_w = mov.shape
    sy = approx_scale_y if approx_scale_y is not None else mov_h / float(ref_h)
    sx = approx_scale_x if approx_scale_x is not None else mov_w / float(ref_w)

    raw_points: list[TiePoint] = []
    ys = np.linspace(template_radius + 1, ref_h - template_radius - 2, max(2, grid_rows))
    xs = np.linspace(template_radius + 1, ref_w - template_radius - 2, max(2, grid_cols))
    ref_edge = _gradient(ref)
    mov_edge = _gradient(mov)
    for y in ys:
        for x in xs:
            point = _match_one_point(
                ref,
                mov,
                ref_edge,
                mov_edge,
                float(y),
                float(x),
                approx_y=float(y * sy + approx_offset_y),
                approx_x=float(x * sx + approx_offset_x),
                template_radius=template_radius,
                search_radius_y=search_radius_y,
                search_radius_x=search_radius_x,
            )
            if point is not None:
                raw_points.append(point)

    kept, rejected = _filter_tie_points(raw_points, min_tie_points=min_tie_points)
    return LocalWarpModel(
        reference_shape=ref.shape,
        moving_shape=mov.shape,
        approx_offset_y=float(approx_offset_y),
        approx_offset_x=float(approx_offset_x),
        approx_scale_y=float(sy),
        approx_scale_x=float(sx),
        tie_points=kept,
        rejected_tie_points=rejected,
    )


def warp_cube_with_model(cube: np.ndarray, model: LocalWarpModel) -> np.ndarray:
    """Warp a moving y,x,band cube to the model reference shape."""

    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
        squeeze = True
    elif arr.ndim == 3:
        squeeze = False
    else:
        raise ValueError("warp_cube_with_model expects 2-D or y,x,band array")
    yy, xx = np.indices(model.reference_shape, dtype=np.float32)
    approx_y = yy * model.approx_scale_y + model.approx_offset_y
    approx_x = xx * model.approx_scale_x + model.approx_offset_x
    dy, dx = interpolate_displacement_field(model, yy, xx)
    sample_y = approx_y + dy
    sample_x = approx_x + dx
    warped = bilinear_sample_cube(arr, sample_y, sample_x)
    return warped[:, :, 0] if squeeze else warped


def interpolate_displacement_field(
    model: LocalWarpModel,
    yy: np.ndarray,
    xx: np.ndarray,
    *,
    k: int = 8,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate tie-point displacements using inverse-distance weighting."""

    if not model.tie_points:
        return np.zeros_like(yy, dtype=np.float32), np.zeros_like(xx, dtype=np.float32)
    points = np.array(
        [[p.ref_y, p.ref_x, p.delta_y, p.delta_x, max(p.score, 0.01)] for p in model.tie_points],
        dtype=np.float32,
    )
    flat_y = yy.reshape(-1)
    flat_x = xx.reshape(-1)
    out_dy = np.empty(flat_y.shape, dtype=np.float32)
    out_dx = np.empty(flat_x.shape, dtype=np.float32)
    k = max(1, min(k, points.shape[0]))
    for start in range(0, flat_y.size, 8192):
        end = min(start + 8192, flat_y.size)
        py = flat_y[start:end][:, None]
        px = flat_x[start:end][:, None]
        dist2 = (py - points[None, :, 0]) ** 2 + (px - points[None, :, 1]) ** 2
        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        d = np.take_along_axis(dist2, idx, axis=1)
        nearest = points[idx]
        weights = 1.0 / np.maximum(d, 1e-4) ** (power * 0.5)
        weights *= nearest[:, :, 4]
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
        out_dy[start:end] = np.sum(weights * nearest[:, :, 2], axis=1)
        out_dx[start:end] = np.sum(weights * nearest[:, :, 3], axis=1)
    return out_dy.reshape(yy.shape), out_dx.reshape(xx.shape)


def bilinear_sample_cube(cube: np.ndarray, sample_y: np.ndarray, sample_x: np.ndarray) -> np.ndarray:
    """Bilinear sample a y,x,band cube at floating source coordinates."""

    arr = np.asarray(cube, dtype=np.float32)
    h, w, bands = arr.shape
    y = np.clip(sample_y.astype(np.float32), 0, h - 1)
    x = np.clip(sample_x.astype(np.float32), 0, w - 1)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    out = np.empty(y.shape + (bands,), dtype=np.float32)
    for b0 in range(0, bands, 32):
        b1 = min(b0 + 32, bands)
        v00 = arr[y0, x0, b0:b1]
        v01 = arr[y0, x1, b0:b1]
        v10 = arr[y1, x0, b0:b1]
        v11 = arr[y1, x1, b0:b1]
        top = v00 * (1.0 - wx[:, :, None]) + v01 * wx[:, :, None]
        bottom = v10 * (1.0 - wx[:, :, None]) + v11 * wx[:, :, None]
        out[:, :, b0:b1] = top * (1.0 - wy[:, :, None]) + bottom * wy[:, :, None]
    return out


def _match_one_point(
    ref: np.ndarray,
    mov: np.ndarray,
    ref_edge: np.ndarray,
    mov_edge: np.ndarray,
    ref_y: float,
    ref_x: float,
    *,
    approx_y: float,
    approx_x: float,
    template_radius: int,
    search_radius_y: int,
    search_radius_x: int,
) -> TiePoint | None:
    r = int(template_radius)
    cy = int(round(ref_y))
    cx = int(round(ref_x))
    ref_patch = _patch(ref, cy, cx, r)
    ref_edge_patch = _patch(ref_edge, cy, cx, r)
    if ref_patch is None or ref_edge_patch is None:
        return None
    best: tuple[int, int, float] | None = None
    ay = int(round(approx_y))
    ax = int(round(approx_x))
    for y in range(ay - search_radius_y, ay + search_radius_y + 1):
        for x in range(ax - search_radius_x, ax + search_radius_x + 1):
            mov_patch = _patch(mov, y, x, r)
            mov_edge_patch = _patch(mov_edge, y, x, r)
            if mov_patch is None or mov_edge_patch is None:
                continue
            score = 0.45 * _ncc(ref_edge_patch, mov_edge_patch) + 0.55 * _nmi(ref_patch, mov_patch)
            if best is None or score > best[2]:
                best = (y, x, float(score))
    if best is None:
        return None
    return TiePoint(
        ref_y=float(cy),
        ref_x=float(cx),
        moving_y=float(best[0]),
        moving_x=float(best[1]),
        approx_y=float(approx_y),
        approx_x=float(approx_x),
        score=float(best[2]),
    )


def _patch(image: np.ndarray, y: int, x: int, r: int) -> np.ndarray | None:
    if y - r < 0 or x - r < 0 or y + r + 1 > image.shape[0] or x + r + 1 > image.shape[1]:
        return None
    return image[y - r : y + r + 1, x - r : x + r + 1]


def _filter_tie_points(
    points: list[TiePoint],
    *,
    min_tie_points: int,
) -> tuple[list[TiePoint], list[TiePoint]]:
    if len(points) <= min_tie_points:
        return points, []
    # Keep useful-score matches first, then remove displacement outliers.
    scores = np.array([p.score for p in points], dtype=np.float32)
    score_floor = max(float(np.percentile(scores, 20)), 0.02)
    candidates = [p for p in points if p.score >= score_floor]
    if len(candidates) < min_tie_points:
        candidates = sorted(points, key=lambda p: p.score, reverse=True)[:min_tie_points]
    dy = np.array([p.delta_y for p in candidates], dtype=np.float32)
    dx = np.array([p.delta_x for p in candidates], dtype=np.float32)
    med_y = float(np.median(dy))
    med_x = float(np.median(dx))
    mad_y = float(np.median(np.abs(dy - med_y))) + 1e-6
    mad_x = float(np.median(np.abs(dx - med_x))) + 1e-6
    kept: list[TiePoint] = []
    rejected: list[TiePoint] = []
    for point in candidates:
        ok_y = abs(point.delta_y - med_y) <= max(3.5 * mad_y, 3.0)
        ok_x = abs(point.delta_x - med_x) <= max(3.5 * mad_x, 2.0)
        (kept if ok_y and ok_x else rejected).append(point)
    if len(kept) < min_tie_points:
        merged = sorted(candidates, key=lambda p: p.score, reverse=True)
        kept = merged[:min_tie_points]
        rejected = merged[min_tie_points:]
    rejected.extend([p for p in points if p not in candidates])
    return kept, rejected


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(arr[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _gradient(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    gy = np.zeros_like(arr)
    gx = np.zeros_like(arr)
    gy[1:-1] = 0.5 * (arr[2:] - arr[:-2])
    gx[:, 1:-1] = 0.5 * (arr[:, 2:] - arr[:, :-2])
    return _normalize_image(np.sqrt(gx * gx + gy * gy))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if valid.sum() < 8:
        return -1.0
    aa = aa[valid] - aa[valid].mean()
    bb = bb[valid] - bb[valid].mean()
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom <= 1e-9:
        return -1.0
    return float(np.sum(aa * bb) / denom)


def _nmi(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    aa = _normalize_image(a).reshape(-1)
    bb = _normalize_image(b).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if valid.sum() < 16:
        return 0.0
    hist2d, _, _ = np.histogram2d(aa[valid], bb[valid], bins=bins, range=[[0, 1], [0, 1]])
    pxy = hist2d / max(float(hist2d.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = _entropy(px)
    hy = _entropy(py)
    hxy = _entropy(pxy.reshape(-1))
    if hxy <= 1e-12:
        return 0.0
    return float((hx + hy) / hxy - 1.0)


def _entropy(p: np.ndarray) -> float:
    values = np.asarray(p, dtype=np.float64)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float32)))


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))
