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
    method: str = "horizontal_gradient_peaks"


def detect_core_boxes(image: np.ndarray, config: M11Config) -> DetectionResult:
    """Detect stacked core-box candidates from a long-strip image."""

    thumbnail, scale_x, scale_y = make_thumbnail(image, config.thumbnail_width)
    orange_boxes = _detect_orange_frame_boxes(thumbnail)
    if orange_boxes is not None:
        raw_h, raw_w = image.shape[:2]
        candidates = []
        thumb_boxes = []
        for index, (bbox_thumb, score) in enumerate(orange_boxes, start=1):
            thumb_boxes.append(bbox_thumb)
            candidates.append(CoreBoxCandidate(
                box_id=f"box_{index:04d}", order_index=index,
                bbox_xyxy_thumb=bbox_thumb,
                bbox_xyxy_raw=_scale_bbox_to_raw(bbox_thumb, scale_x, scale_y, raw_w, raw_h),
                boundary_score=score, split_score=score,
            ))
        return DetectionResult(candidates, thumbnail, thumb_boxes, scale_x, scale_y, [], "orange_frame_pairs")

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


def _detect_orange_frame_boxes(thumbnail: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]] | None:
    """Pair full-width orange top/bottom rails, not horizontal rock fractures."""

    if thumbnail.ndim != 3 or thumbnail.shape[2] < 3:
        return None
    height, width = thumbnail.shape[:2]
    rgb = thumbnail[:, :, :3].astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    orange = (red > 125) & (red > 1.35 * green) & (green > 1.35 * blue) & (red - blue > 80)
    left, right = int(width * 0.12), int(width * 0.88)
    coverage = orange[:, left:right].mean(axis=1)
    active = np.flatnonzero(coverage > 0.35)
    if active.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(active) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, active.size]
    min_run = max(5, width // 25)
    rails = [(int(active[s]), int(active[e - 1]) + 1)
             for s, e in zip(starts, ends)
             if int(active[e - 1]) - int(active[s]) + 1 >= min_run
             and float(coverage[active[s]:active[e - 1] + 1].max()) > 0.65]
    if len(rails) < 4:
        return None
    if len(rails) % 2:
        raise RuntimeError(f"Ambiguous orange box-frame rails ({len(rails)}); review M1-1 thumbnail before cropping")
    centers = [(start + end) / 2 for start, end in rails]
    spans = np.asarray([centers[2 * i + 1] - centers[2 * i] for i in range(len(rails) // 2)])
    gaps = np.asarray([centers[2 * i + 2] - centers[2 * i + 1] for i in range(len(rails) // 2 - 1)])
    typical = float(np.median(spans))
    if (typical < width or np.any(spans < typical * 0.72) or np.any(spans > typical * 1.28)
            or np.any(gaps < 0) or np.any(gaps > typical * 0.28)):
        raise RuntimeError("Orange box-frame pairing is inconsistent; manual M1-1 review is required")

    # The saturated centre of a rail is much narrower than the actual box.
    # Use its less-saturated orange paint to locate the *outer* frame edges.
    weak_orange = (red > 70) & (red > 1.15 * green) & (red - blue > 30)
    pad_x = max(3, int(width * 0.015))
    pad_y = max(3, int(width * 0.012))
    boxes = []
    for index in range(len(rails) // 2):
        top = rails[2 * index]
        bottom = rails[2 * index + 1]
        rail_pixels = np.concatenate((weak_orange[top[0]:top[1]], weak_orange[bottom[0]:bottom[1]]), axis=0)
        frame_x = np.flatnonzero(rail_pixels.mean(axis=0) > 0.30)
        if frame_x.size < width * 0.60:
            raise RuntimeError(f"Orange box {index + 1} has insufficient horizontal frame support")
        x0 = max(0, int(frame_x[0]) - pad_x)
        x1 = min(width - 1, int(frame_x[-1]) + pad_x)
        y0 = max(0, top[0] - pad_y)
        y1 = min(height - 1, bottom[1] + pad_y)
        if index > 0:
            previous_bottom = rails[2 * index - 1][1]
            y0 = max(y0, (previous_bottom + top[0]) // 2)
        if index + 1 < len(rails) // 2:
            next_top = rails[2 * index + 2][0]
            y1 = min(y1, (bottom[1] + next_top) // 2 - 1)
        if y1 <= y0:
            raise RuntimeError(f"Orange box {index + 1} has invalid non-overlapping bounds")
        score = float((coverage[top[0]:top[1]].max() + coverage[bottom[0]:bottom[1]].max()) / 2)
        boxes.append(((x0, y0, x1, y1), score))
    return boxes


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
