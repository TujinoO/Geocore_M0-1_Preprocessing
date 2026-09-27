from __future__ import annotations

import math

import numpy as np

from .config import M11Config
from .image_ops import to_gray
from .models import AngleEstimate


def estimate_box_angle(crop: np.ndarray, inner_bbox_xyxy: tuple[int, int, int, int], config: M11Config) -> AngleEstimate:
    """Estimate correction angle from vertical edge evidence inside a core box."""

    orange_angle = _estimate_angle_by_orange_frame(crop, inner_bbox_xyxy, config)
    if orange_angle is not None:
        return orange_angle

    gray = to_gray(crop)
    h, w = gray.shape
    x0, y0, x1, y1 = inner_bbox_xyxy
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return AngleEstimate(0.0, 0.0, None, None, 0, "fallback")

    if config.angle_scan_enabled:
        projection = _estimate_angle_by_vertical_projection(gray, (x0, y0, x1, y1), config)
        if projection is not None:
            return projection

    return _estimate_angle_by_side_edges(gray, (x0, y0, x1, y1), config)


def _estimate_angle_by_orange_frame(
    crop: np.ndarray, inner_bbox_xyxy: tuple[int, int, int, int], config: M11Config
) -> AngleEstimate | None:
    """Align orange vertical dividers, avoiding misleading rock fractures."""

    if crop.ndim != 3 or crop.shape[2] < 3:
        return None
    frame_sample = crop[::max(1, crop.shape[0] // 2048), ::max(1, crop.shape[1] // 512), :3].astype(np.int16)
    sr, sg, sb = frame_sample[:, :, 0], frame_sample[:, :, 1], frame_sample[:, :, 2]
    frame_orange = (sr > 125) & (sr > 1.35 * sg) & (sg > 1.35 * sb) & (sr - sb > 80)
    left, right = int(frame_sample.shape[1] * 0.10), int(frame_sample.shape[1] * 0.90)
    row_coverage = frame_orange[:, left:right].mean(axis=1)
    band = max(1, int(len(row_coverage) * 0.10))
    if not (np.any(row_coverage[:band] > 0.35) and np.any(row_coverage[-band:] > 0.35)):
        return None
    x0, y0, x1, y1 = inner_bbox_xyxy
    trim = int((y1 - y0 + 1) * 0.09)
    ya, yb = max(0, y0 + trim), min(crop.shape[0], y1 - trim)
    xa, xb = max(0, x0), min(crop.shape[1], x1)
    if yb - ya < 100 or xb - xa < 50:
        return None
    roi = crop[ya:yb, xa:xb, :3].astype(np.int16)
    red, green, blue = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    orange = (red > 125) & (red > 1.35 * green) & (green > 1.35 * blue) & (red - blue > 80)
    ys, xs = np.nonzero(orange)
    if len(xs) < 2000:
        return None
    if len(xs) > config.angle_scan_max_points:
        keep = np.linspace(0, len(xs) - 1, config.angle_scan_max_points).astype(np.int64)
        ys, xs = ys[keep], xs[keep]
    weights = np.ones(len(xs), dtype=np.float64)
    angles = np.arange(-config.angle_scan_range_deg, config.angle_scan_range_deg + 0.025,
                       config.angle_scan_coarse_step_deg, dtype=np.float64)
    coarse = _score_projection_angles(xs, ys, weights, roi.shape[1], roi.shape[0], angles)
    if coarse is None:
        return None
    coarse_angle = coarse[0]
    fine_angles = np.arange(coarse_angle - 0.10, coarse_angle + 0.105,
                            config.angle_scan_fine_step_deg, dtype=np.float64)
    fine = _score_projection_angles(xs, ys, weights, roi.shape[1], roi.shape[0], fine_angles)
    if fine is None or fine[2] < 0.08:
        return None
    angle = float(fine[0] + config.manual_angle_delta_deg)
    return AngleEstimate(angle, min(1.0, fine[2] / 0.15),
                         -math.tan(math.radians(angle)), -math.tan(math.radians(angle)),
                         len(xs), "orange_vertical_frame")


def _estimate_angle_by_side_edges(
    gray: np.ndarray,
    inner_bbox_xyxy: tuple[int, int, int, int],
    config: M11Config,
) -> AngleEstimate:
    x0, y0, x1, y1 = inner_bbox_xyxy
    hgrad = np.zeros_like(gray, dtype=np.float32)
    hgrad[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    margin = max(6, int(config.edge_search_margin_px))

    left_points = _collect_edge_points(hgrad, x0, y0, y1, margin)
    right_points = _collect_edge_points(hgrad, x1, y0, y1, margin)
    left_slope, left_support, left_resid = _robust_line_slope(left_points)
    right_slope, right_support, right_resid = _robust_line_slope(right_points)

    slopes: list[tuple[float, int, float]] = []
    if left_slope is not None:
        slopes.append((left_slope, left_support, left_resid))
    if right_slope is not None:
        slopes.append((right_slope, right_support, right_resid))

    if not slopes:
        return AngleEstimate(0.0, 0.0, None, None, 0, "vertical_edges")

    weights = np.array([max(1, support) / (1.0 + resid) for _, support, resid in slopes], dtype=np.float64)
    values = np.array([slope for slope, _, _ in slopes], dtype=np.float64)
    slope = float(np.average(values, weights=weights))
    angle = -math.degrees(math.atan(slope))
    angle += float(config.manual_angle_delta_deg)
    angle = max(-config.max_abs_angle_deg, min(config.max_abs_angle_deg, angle))

    support_rows = int(sum(support for _, support, _ in slopes))
    expected_rows = max(1, (y1 - y0 + 1) * len(slopes))
    support_ratio = min(1.0, support_rows / expected_rows)
    agreement = 1.0
    if len(slopes) == 2:
        diff = abs(slopes[0][0] - slopes[1][0])
        agreement = max(0.0, 1.0 - diff / 0.08)
    residual_score = max(0.0, 1.0 - float(np.mean([resid for _, _, resid in slopes])) / 4.0)
    confidence = float(max(0.0, min(1.0, 0.55 * support_ratio + 0.30 * agreement + 0.15 * residual_score)))

    return AngleEstimate(
        angle_deg=angle,
        confidence=confidence,
        left_slope=left_slope,
        right_slope=right_slope,
        support_rows=support_rows,
        method="vertical_edges",
    )


def _estimate_angle_by_vertical_projection(
    gray: np.ndarray,
    inner_bbox_xyxy: tuple[int, int, int, int],
    config: M11Config,
) -> AngleEstimate | None:
    """Find the correction angle that best aligns vertical edges into columns.

    The older side-edge estimator can miss the real box wall when the detection
    bbox contains safety background. This projection scan uses all persistent
    vertical evidence in the box: outer walls, separators and long core columns.
    """

    h, w = gray.shape
    x0, y0, x1, y1 = inner_bbox_xyxy
    if x1 <= x0 or y1 <= y0:
        return None

    trim_y = int((y1 - y0 + 1) * 0.06)
    ya = max(0, y0 + trim_y)
    yb = min(h - 1, y1 - trim_y)
    xa = max(0, x0 - 8)
    xb = min(w - 1, x1 + 8)
    if yb <= ya or xb <= xa:
        return None

    roi = gray[ya : yb + 1, xa : xb + 1]
    hgrad = np.zeros_like(roi, dtype=np.float32)
    hgrad[:, 1:-1] = np.abs(roi[:, 2:] - roi[:, :-2])
    threshold = float(np.percentile(hgrad, config.angle_scan_edge_percentile))
    ys, xs = np.nonzero(hgrad >= threshold)
    if len(xs) < 256:
        return None

    weights = hgrad[ys, xs].astype(np.float64)
    max_points = max(1000, int(config.angle_scan_max_points))
    if len(xs) > max_points:
        strongest = np.argpartition(weights, -max_points)[-max_points:]
        ys = ys[strongest]
        xs = xs[strongest]
        weights = weights[strongest]

    coarse_angles = np.arange(
        -float(config.angle_scan_range_deg),
        float(config.angle_scan_range_deg) + 0.5 * float(config.angle_scan_coarse_step_deg),
        float(config.angle_scan_coarse_step_deg),
        dtype=np.float64,
    )
    coarse = _score_projection_angles(xs, ys, weights, roi.shape[1], roi.shape[0], coarse_angles)
    if coarse is None:
        return None
    coarse_angle, coarse_score, _, _ = coarse
    fine_half_width = max(float(config.angle_scan_coarse_step_deg) * 2.0, 0.10)
    fine_start = max(-float(config.angle_scan_range_deg), coarse_angle - fine_half_width)
    fine_end = min(float(config.angle_scan_range_deg), coarse_angle + fine_half_width)
    fine_angles = np.arange(
        fine_start,
        fine_end + 0.5 * float(config.angle_scan_fine_step_deg),
        float(config.angle_scan_fine_step_deg),
        dtype=np.float64,
    )
    fine = _score_projection_angles(xs, ys, weights, roi.shape[1], roi.shape[0], fine_angles)
    if fine is None:
        return None

    angle, score, peak_ratio, concentration = fine
    angle = max(-config.max_abs_angle_deg, min(config.max_abs_angle_deg, float(angle)))
    angle += float(config.manual_angle_delta_deg)
    slope = -math.tan(math.radians(angle))
    confidence = _projection_confidence(score, peak_ratio, concentration, len(xs))
    return AngleEstimate(
        angle_deg=angle,
        confidence=confidence,
        left_slope=slope,
        right_slope=slope,
        support_rows=int(len(xs)),
        method="vertical_edge_projection",
    )


def _score_projection_angles(
    xs: np.ndarray,
    ys: np.ndarray,
    weights: np.ndarray,
    width: int,
    height: int,
    angles_deg: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if len(angles_deg) == 0:
        return None

    y_center = (height - 1) / 2.0
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return None

    best: tuple[float, float, float, float] | None = None
    top_k = max(8, min(24, width // 20))
    for angle in angles_deg:
        slope = -math.tan(math.radians(float(angle)))
        projected = np.rint(xs - slope * (ys - y_center)).astype(np.int32)
        valid = (projected >= 0) & (projected < width)
        if not np.any(valid):
            continue
        hist = np.bincount(projected[valid], weights=weights[valid], minlength=width).astype(np.float64)
        peak_ratio = float(np.sort(hist)[-top_k:].sum() / (hist.sum() + 1e-6))
        concentration = float((hist**2).sum() / (hist.sum() + 1e-6) ** 2)
        score = peak_ratio + 20.0 * concentration
        if best is None or score > best[1]:
            best = (float(angle), float(score), peak_ratio, concentration)
    return best


def _projection_confidence(score: float, peak_ratio: float, concentration: float, point_count: int) -> float:
    point_score = min(1.0, point_count / 50_000.0)
    score_term = min(1.0, max(0.0, (score - 0.08) / 0.14))
    peak_term = min(1.0, max(0.0, (peak_ratio - 0.06) / 0.10))
    concentration_term = min(1.0, max(0.0, concentration / 0.003))
    return float(0.25 * point_score + 0.45 * score_term + 0.20 * peak_term + 0.10 * concentration_term)


def _collect_edge_points(
    hgrad: np.ndarray,
    x_center: int,
    y0: int,
    y1: int,
    margin: int,
) -> np.ndarray:
    h, w = hgrad.shape
    xa = max(1, x_center - margin)
    xb = min(w - 2, x_center + margin)
    if xb <= xa:
        return np.empty((0, 2), dtype=np.float64)
    sub = hgrad[y0 : y1 + 1, xa : xb + 1]
    row_max = sub.max(axis=1)
    threshold = max(float(np.percentile(row_max, 60)), float(np.percentile(hgrad, 75)))
    points: list[tuple[float, float]] = []
    for row_index, value in enumerate(row_max):
        if value < threshold:
            continue
        local_x = int(np.argmax(sub[row_index]))
        points.append((float(y0 + row_index), float(xa + local_x)))
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _robust_line_slope(points: np.ndarray) -> tuple[float | None, int, float]:
    if len(points) < 12:
        return None, 0, float("inf")
    y = points[:, 0]
    x = points[:, 1]
    keep = np.ones(len(points), dtype=bool)
    slope = 0.0
    intercept = float(np.median(x))
    residual_median = float("inf")

    for _ in range(3):
        yy = y[keep]
        xx = x[keep]
        if len(yy) < 12:
            return None, 0, float("inf")
        slope, intercept = np.polyfit(yy, xx, deg=1)
        residual = np.abs(xx - (slope * yy + intercept))
        residual_median = float(np.median(residual))
        cutoff = max(2.5, float(np.percentile(residual, 70)) * 1.8)
        new_keep = np.zeros_like(keep)
        new_keep[np.nonzero(keep)[0][residual <= cutoff]] = True
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return float(slope), int(keep.sum()), residual_median
