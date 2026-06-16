from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from geocore_m1_3.config import LayoutConfig


@dataclass
class Lane:
    lane_id: str
    lane_index: int
    order_index: int
    bbox: list[int]
    centerline: list[list[float]]
    direction: str
    valid_pixel_length: int
    foreground_pixels: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "lane_index": self.lane_index,
            "order_index": self.order_index,
            "bbox": self.bbox,
            "centerline": self.centerline,
            "direction": self.direction,
            "valid_pixel_length": self.valid_pixel_length,
            "foreground_pixels": self.foreground_pixels,
            "confidence": self.confidence,
        }


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values.astype(float), kernel, mode="same")


def _weighted_quantiles(xs: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
    order = np.argsort(xs)
    xs = xs[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    if cdf[-1] <= 0:
        return np.linspace(xs.min(), xs.max(), len(list(quantiles)))
    cdf = cdf / cdf[-1]
    return np.interp(list(quantiles), cdf, xs)


def _weighted_kmeans_1d(weights: np.ndarray, k: int, iterations: int = 50) -> np.ndarray:
    xs = np.arange(len(weights), dtype=float)
    positive = weights > 0
    if positive.sum() < k:
        return np.linspace(0, len(weights) - 1, k)
    q = [(i + 0.5) / k for i in range(k)]
    centers = _weighted_quantiles(xs[positive], weights[positive], q)
    for _ in range(iterations):
        distances = np.abs(xs[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            selected = labels == i
            total = weights[selected].sum()
            if total > 0:
                new_centers[i] = float((xs[selected] * weights[selected]).sum() / total)
        if np.allclose(new_centers, centers, atol=0.1):
            break
        centers = new_centers
    return np.sort(centers)


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev + 1))
        start = prev = value
    runs.append((start, prev + 1))
    return runs


def _direction_for_lane(base_direction: str, allow_snake: bool, lane_index: int) -> str:
    if allow_snake and lane_index % 2 == 0:
        return "bottom_to_top" if base_direction == "top_to_bottom" else "top_to_bottom"
    return base_direction


def detect_lanes(mask: np.ndarray, config: LayoutConfig, core_box_id: str = "core_box") -> list[Lane]:
    if mask.ndim != 2:
        raise ValueError("detect_lanes expects a 2D mask.")
    lane_count = max(1, int(config.lane_count))
    height, width = mask.shape
    projection = mask.sum(axis=0).astype(float)
    smoothed = _smooth(projection, max(5, width // 80))
    centers = _weighted_kmeans_1d(smoothed, lane_count)
    boundaries = [0]
    boundaries.extend(int(round((centers[i] + centers[i + 1]) / 2.0)) for i in range(lane_count - 1))
    boundaries.append(width)

    lanes: list[Lane] = []
    max_projection = max(1.0, float(smoothed.max()))
    for idx in range(lane_count):
        raw_x0 = max(0, boundaries[idx])
        raw_x1 = min(width, boundaries[idx + 1])
        if raw_x1 <= raw_x0:
            continue
        local_projection = projection[raw_x0:raw_x1]
        active_cols = np.flatnonzero(local_projection > max(1.0, max_projection * 0.015))
        if active_cols.size:
            runs = _active_runs(local_projection > max(1.0, max_projection * 0.015))
            # Small false positives near the box edge should not pull the lane
            # boundary outward; use the heaviest continuous foreground run.
            best_start, best_end = max(runs, key=lambda run: float(local_projection[run[0] : run[1]].sum()))
            x0 = raw_x0 + int(best_start) - config.lane_padding_px
            x1 = raw_x0 + int(best_end) + config.lane_padding_px
            x0 = max(0, min(width - 1, x0))
            x1 = max(x0 + 1, min(width, x1))
        else:
            x0, x1 = raw_x0, raw_x1

        lane_mask = mask[:, x0:x1]
        row_counts = lane_mask.sum(axis=1)
        active_rows = np.flatnonzero(row_counts > max(1, (x1 - x0) * 0.01))
        if active_rows.size:
            y0 = max(0, int(active_rows[0]) - config.lane_padding_px)
            y1 = min(height, int(active_rows[-1]) + 1 + config.lane_padding_px)
        else:
            y0, y1 = 0, height

        direction = _direction_for_lane(config.lane_direction, config.allow_snake_order, idx + 1)
        cx = (x0 + x1) / 2.0
        confidence = float(min(1.0, local_projection.sum() / max(1.0, smoothed.sum() / lane_count)))
        lane = Lane(
            lane_id=f"{core_box_id}_L{idx + 1:02d}",
            lane_index=idx + 1,
            order_index=idx + 1,
            bbox=[int(x0), int(y0), int(x1), int(y1)],
            centerline=[[cx, float(y0)], [cx, float(y1)]],
            direction=direction,
            valid_pixel_length=int(max(0, y1 - y0)),
            foreground_pixels=int(lane_mask.sum()),
            confidence=confidence,
        )
        lanes.append(lane)

    if config.lane_order == "right_to_left":
        lanes = list(reversed(lanes))
        for order_index, lane in enumerate(lanes, start=1):
            lane.order_index = order_index
    else:
        for order_index, lane in enumerate(lanes, start=1):
            lane.order_index = order_index
    return lanes
