from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import M11Config
from .image_ops import make_thumbnail, moving_average, to_gray
from .models import CoreBoxCandidate


@dataclass(slots=True)
class BoundaryPeak:
    y: int
    score: float
    coverage: float


@dataclass(slots=True)
class DetectionResult:
    candidates: list[CoreBoxCandidate]
    thumbnail: np.ndarray
    thumbnail_boxes: list[tuple[int, int, int, int]]
    scale_x: float
    scale_y: float
    boundary_peaks: list[BoundaryPeak]


def detect_core_boxes(image: np.ndarray, config: M11Config) -> DetectionResult:
    """Detect stacked core-box candidates from a long-strip image."""

    thumbnail, scale_x, scale_y = make_thumbnail(image, config.thumbnail_width)
    gray = to_gray(thumbnail)
    height, width = gray.shape
    x0_band = int(width * config.horizontal_band_left_ratio)
    x1_band = int(width * config.horizontal_band_right_ratio)
    peaks = _detect_horizontal_boundary_peaks(gray, x0_band, x1_band, config)
    segments = _pair_boundary_peaks(peaks, x1_band - x0_band, config)
    segments = _refine_segment_starts(segments, peaks, x1_band - x0_band, config)

    default_x0 = int(width * config.x_crop_left_ratio)
    default_x1 = int(width * config.x_crop_right_ratio)
    thumb_boxes: list[tuple[int, int, int, int]] = []
    candidates: list[CoreBoxCandidate] = []
    raw_h, raw_w = image.shape[:2]

    for index, (y0, y1, split_score) in enumerate(segments, start=1):
        y0 = max(0, min(height - 1, int(y0)))
        y1 = max(0, min(height - 1, int(y1)))
        if y1 <= y0:
            continue
        x0, x1 = _estimate_x_span(gray, y0, y1, default_x0, default_x1, config)
        bbox_thumb = (x0, y0, x1, y1)
        bbox_raw = _scale_bbox_to_raw(bbox_thumb, scale_x, scale_y, raw_w, raw_h)
        box_id = f"box_{index:04d}"
        thumb_boxes.append(bbox_thumb)
        candidates.append(
            CoreBoxCandidate(
                box_id=box_id,
                order_index=index,
                bbox_xyxy_thumb=bbox_thumb,
                bbox_xyxy_raw=bbox_raw,
                boundary_score=float(split_score),
                split_score=float(split_score),
            )
        )

    return DetectionResult(
        candidates=candidates,
        thumbnail=thumbnail,
        thumbnail_boxes=thumb_boxes,
        scale_x=scale_x,
        scale_y=scale_y,
        boundary_peaks=peaks,
    )


def _detect_horizontal_boundary_peaks(
    gray: np.ndarray,
    x0: int,
    x1: int,
    config: M11Config,
) -> list[BoundaryPeak]:
    central = gray[:, x0:x1]
    vgrad = np.zeros_like(central, dtype=np.float32)
    vgrad[1:-1] = np.abs(central[2:] - central[:-2])
    pixel_threshold = np.percentile(vgrad, config.boundary_pixel_percentile)
    coverage = (vgrad > pixel_threshold).mean(axis=1)
    profile = (
        np.percentile(vgrad, 85, axis=1)
        + 0.5 * vgrad.mean(axis=1)
        + 20.0 * coverage
    )
    profile = moving_average(profile, 5)
    threshold = np.percentile(profile, config.boundary_profile_percentile)
    radius = max(1, int(config.local_peak_radius_px))
    min_distance = max(1, int(config.local_peak_min_distance_px))
    peaks: list[BoundaryPeak] = []

    for y in range(radius, len(profile) - radius):
        value = float(profile[y])
        if value < threshold:
            continue
        window = profile[y - radius : y + radius + 1]
        if value != float(window.max()):
            continue
        peak = BoundaryPeak(y=int(y), score=value, coverage=float(coverage[y]))
        if not peaks or y - peaks[-1].y > min_distance:
            peaks.append(peak)
        elif peak.score > peaks[-1].score:
            peaks[-1] = peak
    return peaks


def _pair_boundary_peaks(
    peaks: list[BoundaryPeak],
    box_width_px: int,
    config: M11Config,
) -> list[tuple[int, int, float]]:
    if len(peaks) < 2:
        return []

    min_h = int(config.min_box_height_to_width * box_width_px)
    max_h = int(config.max_box_height_to_width * box_width_px)
    expected_h = config.expected_box_height_to_width * box_width_px
    segments: list[tuple[int, int, float]] = []
    i = 0
    while i < len(peaks) - 1:
        start = peaks[i]
        candidates: list[tuple[float, int, int]] = []
        for j in range(i + 1, len(peaks)):
            dist = peaks[j].y - start.y
            if dist < min_h:
                continue
            if dist > max_h:
                break
            score = start.score + peaks[j].score - 0.12 * abs(dist - expected_h)
            candidates.append((float(score), j, dist))
        if not candidates:
            i += 1
            continue
        score, end_index, _ = max(candidates, key=lambda item: item[0])
        end = peaks[end_index]
        segments.append((start.y, end.y, score))
        i = end_index
    return segments


def _refine_segment_starts(
    segments: list[tuple[int, int, float]],
    peaks: list[BoundaryPeak],
    box_width_px: int,
    config: M11Config,
) -> list[tuple[int, int, float]]:
    if not segments:
        return []
    min_h = int(config.min_box_height_to_width * box_width_px)
    min_gap = int(config.min_gap_to_refine_start * box_width_px)
    max_gap = int(config.inter_box_gap_to_width * box_width_px)
    median_score = float(np.median([peak.score for peak in peaks])) if peaks else 0.0
    refined: list[tuple[int, int, float]] = []

    for index, (y0, y1, score) in enumerate(segments):
        if index > 0:
            candidates = [
                peak
                for peak in peaks
                if y0 + min_gap <= peak.y <= min(y0 + max_gap, y1 - min_h)
                and peak.score >= median_score
            ]
            if candidates:
                best = max(candidates, key=lambda peak: peak.score + 20.0 * peak.coverage)
                y0 = best.y
        refined.append((int(y0), int(y1), float(score)))
    return refined


def _estimate_x_span(
    gray: np.ndarray,
    y0: int,
    y1: int,
    default_x0: int,
    default_x1: int,
    config: M11Config,
) -> tuple[int, int]:
    """Return a conservative x-span; default ratios keep the whole box when edges are weak."""

    height, width = gray.shape
    y0p = max(0, y0 - 10)
    y1p = min(height - 1, y1 + 10)
    sub = gray[y0p : y1p + 1]
    hgrad = np.zeros_like(sub, dtype=np.float32)
    hgrad[:, 1:-1] = np.abs(sub[:, 2:] - sub[:, :-2])
    profile = np.percentile(hgrad, 86, axis=0) + 0.35 * hgrad.mean(axis=0)
    profile = moving_average(profile, 9)
    search_lo = int(width * 0.14)
    search_hi = int(width * 0.92)
    threshold = np.percentile(profile[search_lo:search_hi], 62)
    peaks: list[tuple[int, float]] = []
    for x in range(search_lo + 4, search_hi - 4):
        value = float(profile[x])
        if value >= threshold and value == float(profile[x - 4 : x + 5].max()):
            peaks.append((x, value))

    best: tuple[float, int, int] | None = None
    min_w = int(width * 0.48)
    max_w = int(width * 0.76)
    expected_w = max(1, default_x1 - default_x0)
    center = width / 2.0
    for left, left_score in peaks:
        for right, right_score in peaks:
            if right <= left:
                continue
            span = right - left
            if span < min_w or span > max_w:
                continue
            span_center = (left + right) / 2.0
            score = (
                left_score
                + right_score
                - 0.12 * abs(span_center - center)
                - 0.03 * abs(span - expected_w)
            )
            if best is None or score > best[0]:
                best = (float(score), int(left), int(right))

    if best is None:
        return int(default_x0), int(default_x1)

    _, left, right = best
    left = min(left, default_x0)
    right = max(right, default_x1)
    left = max(0, left)
    right = min(width - 1, right)
    if right - left < int(width * 0.50):
        return int(default_x0), int(default_x1)
    return int(left), int(right)


def _scale_bbox_to_raw(
    bbox: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    raw_w: int,
    raw_h: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, min(raw_w - 1, int(round(x0 * scale_x)))),
        max(0, min(raw_h - 1, int(round(y0 * scale_y)))),
        max(0, min(raw_w - 1, int(round((x1 + 1) * scale_x)) - 1)),
        max(0, min(raw_h - 1, int(round((y1 + 1) * scale_y)) - 1)),
    )

