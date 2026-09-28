"""Independent RGB evidence for a variable (2-8) physical-slot lattice.

This is a review candidate, not a substitute for field-verified slot labels.
In particular, subharmonic lattices can alias when dividers are obscured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class RgbSlotCandidate:
    count: int
    separator_x: list[int]
    score: float
    runner_up_count: int | None
    score_margin: float
    review_required: bool
    candidates: list[dict]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class DarkRailCandidate:
    """Physical rail proposal from vertically persistent dark RGB columns.

    It is never a reviewed slot label. A regular, high-contrast proposal may
    help an operator place dividers when a U-Net mask bridges adjacent lanes.
    """

    count: int
    separator_x: list[int]
    score: float
    peak_strengths: list[float]
    maximum_spacing_ratio: float
    review_required: bool
    review_reasons: list[str]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(float), (radius, radius), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def estimate_dark_rail_geometry(image_rgb: np.ndarray) -> DarkRailCandidate | None:
    """Suggest 2-8 physical slots from full-height dark tray rails.

    The fixed 768-column normalization makes the minimum peak spacing an
    image-scale rule, not a source-pixel constant. A dark rail is measured
    against a much wider local brightness trend. RGB alone cannot establish
    depth order or prove that every separator is visible, so suspicious edge
    or spacing patterns explicitly require review.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("estimate_dark_rail_geometry expects RGB image")
    height, width = image_rgb.shape[:2]
    if height < 100 or width < 100:
        return None
    rgb = np.asarray(image_rgb[:, :, :3])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    sampled = np.asarray(Image.fromarray(rgb).resize((768, 768), Image.Resampling.BOX), dtype=np.float32)
    gray = sampled[:, :, 0] * 0.299 + sampled[:, :, 1] * 0.587 + sampled[:, :, 2] * 0.114
    column_mean = gray[115:650].mean(axis=0)
    darkness = _gaussian_smooth_1d(column_mean, 35) - _gaussian_smooth_1d(column_mean, 3)
    low, high = np.percentile(darkness, (10, 95))
    darkness = (darkness - low) / max(1e-6, high - low)
    peaks = np.flatnonzero((darkness[1:-1] >= darkness[:-2]) & (darkness[1:-1] > darkness[2:])) + 1
    peaks = peaks[(peaks >= 15) & (peaks <= 760) & (darkness[peaks] >= 0.45)]
    prominent = []
    for peak in peaks:
        left = darkness[max(0, peak - 30):peak]
        right = darkness[peak + 1:min(768, peak + 31)]
        if left.size and right.size and darkness[peak] - max(float(left.min()), float(right.min())) >= 0.08:
            prominent.append(int(peak))
    # Non-maximum suppression merges two dark edges of one thick physical
    # divider, while preserving the narrowest expected 8-slot lattice.
    selected: list[int] = []
    for peak in sorted(prominent, key=lambda value: darkness[value], reverse=True):
        if all(abs(peak - chosen) >= 70 for chosen in selected):
            selected.append(peak)
    selected.sort()
    if len(selected) < 3:
        return None
    count = len(selected) - 1
    gaps = np.diff(selected).astype(float)
    maximum_spacing_ratio = float(gaps.max() / max(1.0, np.median(gaps)))
    reasons = []
    if not 2 <= count <= 8:
        reasons.append("implausible_slot_count")
    # Keep this conservative: loosening both edges to 20%/80% admitted an
    # unboxed negative crop in development QA. Padded corrected boxes may
    # require review even when the count proposal is right.
    if selected[0] > 0.16 * 768 or selected[-1] < 0.84 * 768:
        reasons.append("outer_rail_missing_or_box_crop_incomplete")
    if maximum_spacing_ratio > 1.35:
        reasons.append("irregular_rail_spacing_or_missing_divider")
    strengths = [float(darkness[peak]) for peak in selected]
    score = float(np.mean(strengths) / max(1.0, maximum_spacing_ratio))
    return DarkRailCandidate(
        count=count,
        separator_x=[min(width - 1, max(0, int(round(peak * width / 768)))) for peak in selected],
        score=round(score, 6),
        peak_strengths=[round(value, 6) for value in strengths],
        maximum_spacing_ratio=round(maximum_spacing_ratio, 6),
        review_required=bool(reasons),
        review_reasons=reasons,
    )


def _unit(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (10, 95))
    return np.clip((values - low) / max(1e-6, high - low), 0, 1)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(values, np.ones(width, dtype=float) / width, mode="same")


def estimate_rgb_slot_geometry(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    min_count: int = 2,
    max_count: int = 8,
) -> RgbSlotCandidate | None:
    """Compare regular physical-divider hypotheses without a fixed slot count.

    The scan is capped at roughly 1024x768 samples. RGB evidence uses edges
    persistent along the tray plus a coloured-frame channel; mask evidence is
    complementary and cannot by itself declare an empty slot absent.
    """

    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("estimate_rgb_slot_geometry expects RGB image")
    height, width = image_rgb.shape[:2]
    if height < 100 or width < 100:
        return None
    sy, sx = max(1, height // 1024), max(1, width // 768)
    sampled = image_rgb[::sy, ::sx, :3]
    y0, y1 = int(len(sampled) * 0.12), int(len(sampled) * 0.88)
    rgb = sampled[y0:y1].astype(np.float32)
    if rgb.shape[0] < 30:
        return None
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    derivative = np.zeros_like(gray)
    derivative[:, 2:-2] = np.abs(gray[:, 4:] - gray[:, :-4])
    persistent_edge = np.percentile(derivative, 58, axis=0)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    coloured = (red > 95) & (red > 1.20 * green) & (green > 1.12 * blue) & (red - blue > 45)
    colour_support = coloured.mean(axis=0)
    colour_weight = 0.45 if np.percentile(colour_support, 98) > 0.22 else 0.0
    profile = (1 - colour_weight) * _unit(_smooth(persistent_edge, 5))
    if colour_weight:
        profile += colour_weight * _unit(_smooth(colour_support, 5))
    radius = max(2, len(profile) // 256)
    local = np.asarray([profile[max(0, x - radius):min(len(profile), x + radius + 1)].max()
                        for x in range(len(profile))])
    foreground = None
    if mask is not None:
        if mask.shape != (height, width):
            raise ValueError("RGB and mask dimensions must match")
        foreground = np.asarray(mask[::sy, ::sx][y0:y1], dtype=bool).mean(axis=0)
        foreground = _smooth(foreground, 7)
        foreground /= max(float(foreground.max()), 1e-6)
    sampled_width = len(profile)
    if foreground is not None and foreground.sum() > 1:
        envelope_weights = np.maximum(0, foreground - 0.08)
    else:
        brightness = _unit(_smooth(np.percentile(gray, 55, axis=0), 9))
        texture = np.abs(gray[3:] - gray[:-3]).mean(axis=0)
        envelope_weights = np.maximum(0, 0.65 * brightness + 0.35 * _unit(_smooth(texture, 9)) - 0.28)
    cumulative = np.cumsum(envelope_weights)
    if cumulative[-1] <= 0:
        return None
    envelope_left = int(np.searchsorted(cumulative, cumulative[-1] * 0.025))
    envelope_right = int(np.searchsorted(cumulative, cumulative[-1] * 0.975))
    envelope_tolerance = max(8, int(sampled_width * 0.11))
    results = []
    for count in range(min_count, max_count + 1):
        best = None
        min_spacing = max(12, int(sampled_width * 0.22 / count))
        max_spacing = max(min_spacing, int(sampled_width * 0.92 / count))
        for spacing in range(min_spacing, max_spacing + 1, 2):
            span = count * spacing
            if span < sampled_width * 0.22 or span > sampled_width * 0.92:
                continue
            firsts = np.arange(max(2, int(sampled_width * 0.02)),
                                min(int(sampled_width * 0.72), sampled_width - span - 2), 2)
            firsts = firsts[(np.abs(firsts - envelope_left) <= envelope_tolerance)
                            & (np.abs(firsts + span - envelope_right) <= envelope_tolerance)]
            if not len(firsts):
                continue
            lines = firsts[:, None] + spacing * np.arange(count + 1)[None, :]
            fractions = np.asarray((0.25, 1 / 3, 0.5, 2 / 3, 0.75))
            inside = firsts[:, None, None] + spacing * (
                np.arange(count)[None, :, None] + fractions[None, None, :]
            )
            inside = inside.astype(int)
            middle = (firsts[:, None] + spacing * (np.arange(count)[None, :] + 0.5)).astype(int)
            line_support = local[lines]
            score = 0.60 * line_support.mean(axis=1) + 0.30 * line_support.min(axis=1)
            # A two-slot subharmonic of a six-slot tray can hit every third
            # physical divider. Inspect the whole bay, not only its midpoint.
            score -= 0.30 * local[inside].max(axis=2).mean(axis=1)
            if foreground is not None:
                score += 0.10 * foreground[middle].mean(axis=1)
                score -= 0.14 * foreground[lines].mean(axis=1)
            # Avoid fitting the belt's far edges instead of the central tray.
            score -= 0.08 * np.abs((firsts + span / 2) - sampled_width / 2) / sampled_width
            chosen = int(score.argmax())
            if best is None or score[chosen] > best[0]:
                best = (float(score[chosen]), lines[chosen].copy(),
                        float(line_support[chosen].min()))
        if best is not None:
            score, lines, weakest = best
            results.append({"count": count, "separator_x": [min(width - 1, int(x * sx)) for x in lines],
                            "score": round(score, 6), "weakest_separator": round(weakest, 6)})
    if not results:
        return None
    results.sort(key=lambda item: item["score"], reverse=True)
    top = results[0]
    runner = results[1] if len(results) > 1 else None
    margin = top["score"] - runner["score"] if runner else top["score"]
    return RgbSlotCandidate(
        count=top["count"], separator_x=top["separator_x"], score=top["score"],
        runner_up_count=runner["count"] if runner else None,
        score_margin=round(margin, 6),
        review_required=(mask is None or top["score"] < 0.42
                         or top["weakest_separator"] < 0.40 or margin < 0.06),
        candidates=results,
    )
