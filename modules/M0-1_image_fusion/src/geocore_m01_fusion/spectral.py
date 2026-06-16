"""Spectral calibration and stitching utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .resample import resize_cube


@dataclass(slots=True)
class StitchResult:
    nir: np.ndarray
    swir: np.ndarray
    wavelengths: list[float]
    band_metadata: list[dict[str, Any]]
    stitch_model: dict[str, Any]


def default_wavelengths(count: int, start: float = 0.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(int(count))]


def calibrate_and_concat(
    nir: np.ndarray,
    swir: np.ndarray,
    nir_wavelengths: list[float] | None,
    swir_wavelengths: list[float] | None,
    *,
    mask: np.ndarray | None = None,
) -> StitchResult:
    """Calibrate SWIR scale to NIR overlap and concatenate both sensors.

    The first implementation keeps all original bands. Overlap metadata records
    matching weights so downstream code can later resample to a unified spectral
    grid without losing traceability.
    """

    nir_arr = np.asarray(nir, dtype=np.float32)
    swir_arr = np.asarray(swir, dtype=np.float32)
    if nir_arr.ndim != 3 or swir_arr.ndim != 3:
        raise ValueError("NIR and SWIR inputs must be y,x,band cubes")

    nir_w = nir_wavelengths or default_wavelengths(nir_arr.shape[2])
    swir_w = swir_wavelengths or default_wavelengths(swir_arr.shape[2])
    gain, offset, pairs = _estimate_swir_to_nir(nir_arr, swir_arr, nir_w, swir_w, mask)
    swir_cal = swir_arr * gain + offset

    wavelengths = [float(w) for w in nir_w] + [float(w) for w in swir_w]
    overlap_min = max(min(nir_w), min(swir_w)) if nir_w and swir_w else None
    overlap_max = min(max(nir_w), max(swir_w)) if nir_w and swir_w else None

    band_metadata: list[dict[str, Any]] = []
    for idx, w in enumerate(nir_w):
        in_overlap = overlap_min is not None and overlap_min <= w <= overlap_max
        band_metadata.append(
            {
                "fused_band_index": idx,
                "wavelength_nm": float(w),
                "fwhm_nm": "",
                "source_sensor": "NIR",
                "source_band_index": idx,
                "source_wavelength_nm": float(w),
                "overlap_weight_nir": 1.0 if in_overlap else "",
                "overlap_weight_swir": 0.0 if in_overlap else "",
                "is_bad_band": False,
                "bad_reason": "",
                "snr_estimate": "",
                "recommended_for_mineral": True,
                "recommended_for_visualization": True,
            }
        )
    offset_idx = len(band_metadata)
    for idx, w in enumerate(swir_w):
        in_overlap = overlap_min is not None and overlap_min <= w <= overlap_max
        band_metadata.append(
            {
                "fused_band_index": offset_idx + idx,
                "wavelength_nm": float(w),
                "fwhm_nm": "",
                "source_sensor": "SWIR_CALIBRATED",
                "source_band_index": idx,
                "source_wavelength_nm": float(w),
                "overlap_weight_nir": 0.0 if in_overlap else "",
                "overlap_weight_swir": 1.0 if in_overlap else "",
                "is_bad_band": False,
                "bad_reason": "",
                "snr_estimate": "",
                "recommended_for_mineral": True,
                "recommended_for_visualization": True,
            }
        )

    stitch_model = {
        "type": "global_linear_swir_to_nir",
        "swir_gain": float(gain),
        "swir_offset": float(offset),
        "overlap_pair_count": len(pairs),
        "overlap_pairs": pairs[:64],
        "output_policy": "concat_preserve_all_bands",
    }
    return StitchResult(
        nir=nir_arr,
        swir=swir_cal.astype(np.float32, copy=False),
        wavelengths=wavelengths,
        band_metadata=band_metadata,
        stitch_model=stitch_model,
    )


def concat_upsampled(stitched: StitchResult, target_shape: tuple[int, int], method: str) -> np.ndarray:
    """Resize calibrated NIR and SWIR to target shape and concatenate."""

    nir_high = resize_cube(stitched.nir, target_shape, method=method)
    swir_high = resize_cube(stitched.swir, target_shape, method=method)
    return np.concatenate([nir_high, swir_high], axis=2).astype(np.float32, copy=False)


def _estimate_swir_to_nir(
    nir: np.ndarray,
    swir: np.ndarray,
    nir_w: list[float],
    swir_w: list[float],
    mask: np.ndarray | None,
) -> tuple[float, float, list[dict[str, float]]]:
    pairs = _nearest_overlap_pairs(nir_w, swir_w)
    if not pairs:
        return 1.0, 0.0, []

    swir_to_nir_shape = resize_cube(swir[:, :, [j for _, j in pairs]], nir.shape[:2], method="bilinear")
    nir_overlap = nir[:, :, [i for i, _ in pairs]]
    a = swir_to_nir_shape.reshape(-1)
    b = nir_overlap.reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != nir.shape[:2]:
            m = resize_cube(m, nir.shape[:2], method="nearest")
        valid &= np.repeat((m.reshape(-1) > 0.5), len(pairs))
    if valid.sum() < 32:
        return 1.0, 0.0, _pairs_to_dicts(pairs, nir_w, swir_w)

    a = a[valid]
    b = b[valid]
    # Robust percentile trimming prevents saturated pixels dominating the scale.
    lo_a, hi_a = np.percentile(a, [2, 98])
    lo_b, hi_b = np.percentile(b, [2, 98])
    keep = (a >= lo_a) & (a <= hi_a) & (b >= lo_b) & (b <= hi_b)
    a = a[keep]
    b = b[keep]
    if a.size < 32 or float(np.var(a)) <= 1e-10:
        return 1.0, float(np.nanmedian(b) - np.nanmedian(a)), _pairs_to_dicts(pairs, nir_w, swir_w)
    cov = float(np.mean((a - a.mean()) * (b - b.mean())))
    gain = cov / float(np.var(a))
    offset = float(b.mean() - gain * a.mean())
    if not np.isfinite(gain) or abs(gain) > 10:
        gain = 1.0
    if not np.isfinite(offset) or abs(offset) > 10:
        offset = 0.0
    return float(gain), float(offset), _pairs_to_dicts(pairs, nir_w, swir_w)


def _nearest_overlap_pairs(nir_w: list[float], swir_w: list[float]) -> list[tuple[int, int]]:
    if not nir_w or not swir_w:
        return []
    lo = max(min(nir_w), min(swir_w))
    hi = min(max(nir_w), max(swir_w))
    if lo > hi:
        return []
    sw = np.asarray(swir_w, dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    sw_step = float(np.median(np.abs(np.diff(sw)))) if len(sw) > 1 else 1.0
    for i, w in enumerate(nir_w):
        if not (lo <= w <= hi):
            continue
        j = int(np.argmin(np.abs(sw - float(w))))
        if abs(float(sw[j]) - float(w)) <= max(2.0 * sw_step, 1.0):
            pairs.append((i, j))
    # Keep a bounded set of well-spread pairs for speed.
    if len(pairs) > 64:
        idx = np.linspace(0, len(pairs) - 1, 64).round().astype(int)
        pairs = [pairs[int(k)] for k in idx]
    return pairs


def _pairs_to_dicts(
    pairs: list[tuple[int, int]],
    nir_w: list[float],
    swir_w: list[float],
) -> list[dict[str, float]]:
    return [
        {
            "nir_band_index": int(i),
            "nir_wavelength_nm": float(nir_w[i]),
            "swir_band_index": int(j),
            "swir_wavelength_nm": float(swir_w[j]),
        }
        for i, j in pairs
    ]

