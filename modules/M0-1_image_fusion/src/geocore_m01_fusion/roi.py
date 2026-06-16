"""Registration preview and aligned ROI extraction for raw ENVI triplets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .envi import EnviMetadata, read_envi, write_envi
from .output import write_json
from .preprocess import normalize_rgb
from .registration import register_local_warp, warp_cube_with_model
from .resample import resize_cube


@dataclass(slots=True)
class SensorRegistration:
    sensor: str
    source_shape: tuple[int, int]
    rgb_shape: tuple[int, int]
    scale_x: float
    scale_y: float
    translate_x_rgb: float
    translate_y_rgb: float
    preview_shift_x: float
    preview_shift_y: float
    confidence: float
    method: str

    def source_to_rgb(self, y: np.ndarray | float, x: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
        return y * self.scale_y + self.translate_y_rgb, x * self.scale_x + self.translate_x_rgb

    def rgb_to_source(self, y: np.ndarray | float, x: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
        return (y - self.translate_y_rgb) / self.scale_y, (x - self.translate_x_rgb) / self.scale_x

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "source_shape": [int(x) for x in self.source_shape],
            "rgb_shape": [int(x) for x in self.rgb_shape],
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "translate_x_rgb": self.translate_x_rgb,
            "translate_y_rgb": self.translate_y_rgb,
            "preview_shift_x": self.preview_shift_x,
            "preview_shift_y": self.preview_shift_y,
            "confidence": self.confidence,
            "method": self.method,
        }


def prepare_aligned_roi(
    root: str | Path,
    output_dir: str | Path,
    *,
    crop_height: int = 768,
    crop_width: int = 512,
    preview_width: int = 256,
    max_preview_height: int = 3072,
    registration_mode: str = "scale_only",
    local_refine: bool = True,
    anchor_nir_to_swir: bool = True,
    refine_anchored_nir: bool = True,
    local_warp: bool = True,
    final_hsi_rgb_warp: bool = True,
) -> dict[str, Any]:
    """Register RGB/NIR/SWIR by preview phase correlation and extract a test ROI."""

    root = Path(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    (output / "aligned_envi").mkdir(exist_ok=True)

    paths = _find_triplet(root)
    rgb, rgb_meta = read_envi(paths["rgb_hdr"], paths["rgb_dat"], mmap=True)
    nir, nir_meta = read_envi(paths["nir_hdr"], paths["nir_dat"], mmap=True)
    swir, swir_meta = read_envi(paths["swir_hdr"], paths["swir_dat"], mmap=True)

    rgb_preview, preview_scale_y, preview_scale_x = _rgb_preview(rgb, preview_width, max_preview_height)
    nir_struct = _hsi_structure(nir, bands=_select_bands(nir_meta, [750, 900, 1100, 1300]))
    swir_struct = _hsi_structure(swir, bands=_select_bands(swir_meta, [1200, 1600, 2200, 2350]))
    nir_preview = resize_cube(nir_struct, rgb_preview.shape, method="bilinear")
    swir_preview = resize_cube(swir_struct, rgb_preview.shape, method="bilinear")

    nir_reg = _estimate_registration(
        "NIR",
        nir.shape[:2],
        rgb.shape[:2],
        rgb_preview,
        nir_preview,
        preview_scale_y,
        preview_scale_x,
        registration_mode=registration_mode,
    )
    swir_reg = _estimate_registration(
        "SWIR",
        swir.shape[:2],
        rgb.shape[:2],
        rgb_preview,
        swir_preview,
        preview_scale_y,
        preview_scale_x,
        registration_mode=registration_mode,
    )
    crop = _choose_rgb_crop(rgb_preview, rgb.shape[:2], [nir_reg, swir_reg], crop_height, crop_width, preview_scale_y, preview_scale_x)

    rgb_crop = np.asarray(rgb[crop["rgb_y0"] : crop["rgb_y1"], crop["rgb_x0"] : crop["rgb_x1"], :3])
    swir_crop, swir_source_window = _crop_source_for_rgb_window(
        swir,
        swir_reg,
        crop,
        source_structure=swir_struct,
        rgb_crop=rgb_crop,
        local_refine=local_refine,
    )
    if anchor_nir_to_swir:
        nir_crop, nir_source_window = _crop_source_from_anchor_window(
            nir,
            nir_struct,
            swir_struct,
            swir_source_window,
            anchor_shape=swir.shape[:2],
            local_refine=local_refine,
            refine_with_anchor=refine_anchored_nir,
        )
    else:
        nir_crop, nir_source_window = _crop_source_for_rgb_window(
            nir,
            nir_reg,
            crop,
            source_structure=nir_struct,
            rgb_crop=rgb_crop,
            local_refine=local_refine,
        )
    warp_models: dict[str, Any] = {}
    if local_warp:
        rgb_low_for_swir = _rgb_gray_resized(rgb_crop, swir_crop.shape[:2])
        swir_struct_crop = _normalize_image(np.nanmean(np.asarray(swir_crop, dtype=np.float32), axis=2))
        swir_model = register_local_warp(
            rgb_low_for_swir,
            swir_struct_crop,
            grid_rows=7,
            grid_cols=5,
            template_radius=max(5, min(10, min(swir_crop.shape[:2]) // 8)),
            search_radius_y=max(8, swir_crop.shape[0] // 10),
            search_radius_x=max(4, swir_crop.shape[1] // 10),
            min_tie_points=8,
        )
        swir_crop = warp_cube_with_model(swir_crop, swir_model)
        warp_models["swir_to_rgb_lowres"] = swir_model.to_dict()

        swir_ref = _normalize_image(np.nanmean(np.asarray(swir_crop, dtype=np.float32), axis=2))
        nir_struct_crop = _normalize_image(np.nanmean(np.asarray(nir_crop, dtype=np.float32), axis=2))
        nir_model = register_local_warp(
            swir_ref,
            nir_struct_crop,
            approx_scale_y=nir_struct_crop.shape[0] / float(swir_ref.shape[0]),
            approx_scale_x=nir_struct_crop.shape[1] / float(swir_ref.shape[1]),
            grid_rows=7,
            grid_cols=5,
            template_radius=max(5, min(10, min(swir_ref.shape[:2]) // 8)),
            search_radius_y=max(10, nir_struct_crop.shape[0] // 8),
            search_radius_x=max(5, nir_struct_crop.shape[1] // 8),
            min_tie_points=8,
        )
        nir_crop = warp_cube_with_model(nir_crop, nir_model)
        warp_models["nir_to_swir_lowres"] = nir_model.to_dict()

        if final_hsi_rgb_warp:
            hsi_refined_structure = _joint_hsi_structure(nir_crop, swir_crop)
            rgb_low_for_hsi = _rgb_gray_resized(rgb_crop, hsi_refined_structure.shape)
            final_radius = max(4, min(8, min(hsi_refined_structure.shape[:2]) // 10))
            final_model = register_local_warp(
                rgb_low_for_hsi,
                hsi_refined_structure,
                grid_rows=9,
                grid_cols=6,
                template_radius=final_radius,
                search_radius_y=max(6, min(14, hsi_refined_structure.shape[0] // 10)),
                search_radius_x=max(4, min(8, hsi_refined_structure.shape[1] // 10)),
                min_tie_points=10,
            )
            final_model.method = "joint_hsi_to_rgb_lowres_final_grid_tie_points_idw_warp"
            if _warp_model_is_usable(final_model, min_tie_points=10, min_score=0.08):
                swir_crop = warp_cube_with_model(swir_crop, final_model)
                nir_crop = warp_cube_with_model(nir_crop, final_model)
                final_info = final_model.to_dict()
                final_info["applied"] = True
            else:
                final_info = final_model.to_dict()
                final_info["applied"] = False
                final_info["skip_reason"] = "insufficient_or_low_confidence_tie_points"
            warp_models["joint_hsi_to_rgb_lowres_final"] = final_info

    rgb_hdr, rgb_dat = write_envi(
        rgb_crop,
        output / "aligned_envi" / "RGB_aligned_roi.hdr",
        output / "aligned_envi" / "RGB_aligned_roi.dat",
        interleave="bil",
        description="RGB aligned ROI for M0-1 testing",
    )
    nir_hdr, nir_dat = write_envi(
        nir_crop.astype(np.float32, copy=False),
        output / "aligned_envi" / "NIR_aligned_roi.hdr",
        output / "aligned_envi" / "NIR_aligned_roi.dat",
        wavelengths=nir_meta.wavelengths,
        interleave="bil",
        description="NIR aligned ROI for M0-1 testing",
    )
    swir_hdr, swir_dat = write_envi(
        swir_crop.astype(np.float32, copy=False),
        output / "aligned_envi" / "SWIR_aligned_roi.hdr",
        output / "aligned_envi" / "SWIR_aligned_roi.dat",
        wavelengths=swir_meta.wavelengths,
        interleave="bil",
        description="SWIR aligned ROI for M0-1 testing",
    )

    _save_preview(rgb_crop, output / "previews" / "roi_rgb.png")
    _save_registered_sensor_preview(nir_crop, rgb_crop.shape[:2], output / "previews" / "roi_nir_registered.png")
    _save_registered_sensor_preview(swir_crop, rgb_crop.shape[:2], output / "previews" / "roi_swir_registered.png")
    _save_overlay(rgb_crop, nir_crop, output / "previews" / "roi_overlay_rgb_nir.png")
    _save_overlay(rgb_crop, swir_crop, output / "previews" / "roi_overlay_rgb_swir.png")
    _save_hsi_overlay(nir_crop, swir_crop, output / "previews" / "roi_overlay_nir_swir.png")
    _save_global_previews(rgb_preview, nir_preview, swir_preview, crop, preview_scale_y, preview_scale_x, output / "previews")

    result = {
        "schema_version": "m0_aligned_roi.v1",
        "source_root": str(root),
        "output_dir": str(output),
        "rgb_shape": [int(x) for x in rgb.shape[:2]],
        "nir_shape": [int(x) for x in nir.shape[:2]],
        "swir_shape": [int(x) for x in swir.shape[:2]],
        "registrations": {
            "nir_to_rgb": nir_reg.to_dict(),
            "swir_to_rgb": swir_reg.to_dict(),
        },
        "registration_mode": registration_mode,
        "local_refine": local_refine,
        "anchor_nir_to_swir": anchor_nir_to_swir,
        "refine_anchored_nir": refine_anchored_nir,
        "local_warp": local_warp,
        "final_hsi_rgb_warp": final_hsi_rgb_warp,
        "crop_rgb_window": crop,
        "source_windows": {
            "nir": nir_source_window,
            "swir": swir_source_window,
        },
        "warp_models": warp_models,
        "outputs": {
            "rgb_hdr": str(rgb_hdr),
            "rgb_dat": str(rgb_dat),
            "nir_hdr": str(nir_hdr),
            "nir_dat": str(nir_dat),
            "swir_hdr": str(swir_hdr),
            "swir_dat": str(swir_dat),
            "previews": {
                "rgb": str(output / "previews" / "roi_rgb.png"),
                "nir_registered": str(output / "previews" / "roi_nir_registered.png"),
                "swir_registered": str(output / "previews" / "roi_swir_registered.png"),
                "overlay_rgb_nir": str(output / "previews" / "roi_overlay_rgb_nir.png"),
                "overlay_rgb_swir": str(output / "previews" / "roi_overlay_rgb_swir.png"),
                "overlay_nir_swir": str(output / "previews" / "roi_overlay_nir_swir.png"),
            },
        },
        "notes": [
            "Registration is estimated from downsampled structural previews.",
            "When enabled, the final joint HSI-to-RGB low-resolution warp is applied identically to NIR and SWIR to preserve their internal alignment.",
            "The cropped NIR/SWIR files remain in their native lower resolution but correspond to the RGB crop window via the recorded transforms.",
            "Use these aligned ROI files for small-sample fusion tests before running full-size streaming fusion.",
        ],
    }
    write_json(result, output / "roi_manifest.json")
    return result


def _find_triplet(root: Path) -> dict[str, Path]:
    files = {p.name.upper(): p for p in root.iterdir() if p.is_file()}
    out: dict[str, Path] = {}
    for key, prefix in [("rgb", "RGB-"), ("nir", "NIR-"), ("swir", "SWIR-")]:
        hdrs = [p for p in root.glob(f"{prefix}*.hdr")]
        dats = [p for p in root.glob(f"{prefix}*.dat")]
        if not hdrs or not dats:
            raise FileNotFoundError(f"Cannot find {prefix}*.hdr/.dat under {root}")
        out[f"{key}_hdr"] = hdrs[0]
        out[f"{key}_dat"] = dats[0]
    return out


def _rgb_preview(rgb: np.ndarray, preview_width: int, max_preview_height: int) -> tuple[np.ndarray, float, float]:
    h, w = rgb.shape[:2]
    sx = max(1, int(round(w / preview_width)))
    sy = max(1, int(round((h / w) * preview_width / max_preview_height * sx)))
    # Keep the preview tall enough for line-scan registration but bounded.
    while int(np.ceil(h / sy)) > max_preview_height:
        sy += 1
    sample = np.asarray(rgb[::sy, ::sx, :3])
    lum = normalize_rgb(sample)
    gray = 0.2126 * lum[:, :, 0] + 0.7152 * lum[:, :, 1] + 0.0722 * lum[:, :, 2]
    return _normalize_image(gray), float(sy), float(sx)


def _hsi_structure(cube: np.ndarray, bands: list[int]) -> np.ndarray:
    arr = np.asarray(cube[:, :, bands], dtype=np.float32)
    img = np.nanmean(arr, axis=2)
    return _normalize_image(img)


def _select_bands(meta: EnviMetadata, targets: list[float]) -> list[int]:
    if not meta.wavelengths:
        idx = np.linspace(0, meta.bands - 1, min(4, meta.bands)).round().astype(int).tolist()
        return sorted(set(idx))
    waves = np.asarray(meta.wavelengths, dtype=np.float32)
    idx = [int(np.argmin(np.abs(waves - target))) for target in targets]
    return sorted(set(i for i in idx if 0 <= i < meta.bands))


def _estimate_registration(
    sensor: str,
    source_shape: tuple[int, int],
    rgb_shape: tuple[int, int],
    rgb_preview: np.ndarray,
    source_preview: np.ndarray,
    preview_scale_y: float,
    preview_scale_x: float,
    registration_mode: str,
) -> SensorRegistration:
    dy, dx, confidence = _phase_shift(_gradient(rgb_preview), _gradient(source_preview))
    scale_y = rgb_shape[0] / float(source_shape[0])
    scale_x = rgb_shape[1] / float(source_shape[1])
    if registration_mode == "scale_only":
        return SensorRegistration(
            sensor=sensor,
            source_shape=source_shape,
            rgb_shape=rgb_shape,
            scale_x=scale_x,
            scale_y=scale_y,
            translate_x_rgb=0.0,
            translate_y_rgb=0.0,
            preview_shift_x=float(dx),
            preview_shift_y=float(dy),
            confidence=float(confidence),
            method="scale_only_phase_diagnostic",
        )
    if registration_mode != "phase":
        raise ValueError("registration_mode must be 'scale_only' or 'phase'")
    max_y = rgb_preview.shape[0] * 0.08
    max_x = rgb_preview.shape[1] * 0.08
    method = "phase_correlation_gradient"
    if abs(dy) > max_y or abs(dx) > max_x or confidence < 4.0:
        dy = 0.0
        dx = 0.0
        method = "scale_only_fallback"
    return SensorRegistration(
        sensor=sensor,
        source_shape=source_shape,
        rgb_shape=rgb_shape,
        scale_x=scale_x,
        scale_y=scale_y,
        translate_x_rgb=float(dx * preview_scale_x),
        translate_y_rgb=float(dy * preview_scale_y),
        preview_shift_x=float(dx),
        preview_shift_y=float(dy),
        confidence=float(confidence),
        method=method,
    )


def _phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    ref = reference.astype(np.float32)
    mov = moving.astype(np.float32)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    mov = (mov - mov.mean()) / (mov.std() + 1e-6)
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = float(peak[0])
    dx = float(peak[1])
    if dy > corr.shape[0] / 2:
        dy -= corr.shape[0]
    if dx > corr.shape[1] / 2:
        dx -= corr.shape[1]
    confidence = float((corr[peak] - np.median(corr)) / (np.std(corr) + 1e-6))
    return dy, dx, confidence


def _choose_rgb_crop(
    rgb_preview: np.ndarray,
    rgb_shape: tuple[int, int],
    regs: list[SensorRegistration],
    crop_height: int,
    crop_width: int,
    preview_scale_y: float,
    preview_scale_x: float,
) -> dict[str, int | float]:
    y_min, x_min = 0.0, 0.0
    y_max, x_max = float(rgb_shape[0]), float(rgb_shape[1])
    for reg in regs:
        sy, sx = reg.source_shape
        top, left = reg.source_to_rgb(0.0, 0.0)
        bottom, right = reg.source_to_rgb(float(sy - 1), float(sx - 1))
        y_min = max(y_min, min(top, bottom))
        x_min = max(x_min, min(left, right))
        y_max = min(y_max, max(top, bottom))
        x_max = min(x_max, max(left, right))

    crop_height = min(crop_height, int(y_max - y_min))
    crop_width = min(crop_width, int(x_max - x_min))
    if crop_height < 64 or crop_width < 64:
        raise ValueError("Common registered coverage is too small for ROI extraction")

    py0 = max(0, int(np.floor(y_min / preview_scale_y)))
    py1 = min(rgb_preview.shape[0], int(np.ceil(y_max / preview_scale_y)))
    px0 = max(0, int(np.floor(x_min / preview_scale_x)))
    px1 = min(rgb_preview.shape[1], int(np.ceil(x_max / preview_scale_x)))
    wh = max(4, int(round(crop_height / preview_scale_y)))
    ww = max(4, int(round(crop_width / preview_scale_x)))
    score = _gradient(rgb_preview) * _valid_brightness(rgb_preview)
    best = _best_window(score, py0, py1, px0, px1, wh, ww)
    cy = int(round(best[0] * preview_scale_y))
    cx = int(round(best[1] * preview_scale_x))
    cy = int(np.clip(cy, int(y_min), int(y_max) - crop_height))
    cx = int(np.clip(cx, int(x_min), int(x_max) - crop_width))
    return {
        "rgb_y0": cy,
        "rgb_y1": cy + crop_height,
        "rgb_x0": cx,
        "rgb_x1": cx + crop_width,
        "height": crop_height,
        "width": crop_width,
        "common_y_min": float(y_min),
        "common_y_max": float(y_max),
        "common_x_min": float(x_min),
        "common_x_max": float(x_max),
    }


def _best_window(score: np.ndarray, y0: int, y1: int, x0: int, x1: int, wh: int, ww: int) -> tuple[int, int]:
    y1 = max(y0 + wh, y1)
    x1 = max(x0 + ww, x1)
    sub = score[y0:y1, x0:x1]
    if sub.shape[0] <= wh or sub.shape[1] <= ww:
        return y0, x0
    integ = np.pad(sub, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    sums = integ[wh:, ww:] - integ[:-wh, ww:] - integ[wh:, :-ww] + integ[:-wh, :-ww]
    iy, ix = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return y0 + int(iy), x0 + int(ix)


def _crop_source_for_rgb_window(
    source: np.ndarray,
    reg: SensorRegistration,
    crop: dict[str, int | float],
    *,
    source_structure: np.ndarray | None = None,
    rgb_crop: np.ndarray | None = None,
    local_refine: bool = False,
) -> tuple[np.ndarray, dict[str, int]]:
    sy0f, sx0f = reg.rgb_to_source(float(crop["rgb_y0"]), float(crop["rgb_x0"]))
    sy1f, sx1f = reg.rgb_to_source(float(crop["rgb_y1"]), float(crop["rgb_x1"]))
    sy0 = int(np.clip(np.floor(min(sy0f, sy1f)), 0, source.shape[0] - 1))
    sy1 = int(np.clip(np.ceil(max(sy0f, sy1f)), sy0 + 1, source.shape[0]))
    sx0 = int(np.clip(np.floor(min(sx0f, sx1f)), 0, source.shape[1] - 1))
    sx1 = int(np.clip(np.ceil(max(sx0f, sx1f)), sx0 + 1, source.shape[1]))
    refine_info: dict[str, int | float | bool] = {"applied": False}
    if local_refine and source_structure is not None and rgb_crop is not None:
        refined = _refine_source_window(source_structure, rgb_crop, sy0, sy1, sx0, sx1)
        sy0, sy1, sx0, sx1, refine_info = refined
    return np.asarray(source[sy0:sy1, sx0:sx1, :]), {
        "source_y0": sy0,
        "source_y1": sy1,
        "source_x0": sx0,
        "source_x1": sx1,
        "height": sy1 - sy0,
        "width": sx1 - sx0,
        "local_refine": refine_info,
    }


def _crop_source_from_anchor_window(
    source: np.ndarray,
    source_structure: np.ndarray,
    anchor_structure: np.ndarray,
    anchor_window: dict[str, int | float | dict[str, int | float | bool]],
    *,
    anchor_shape: tuple[int, int],
    local_refine: bool,
    refine_with_anchor: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    src_h, src_w = source.shape[:2]
    anchor_h, anchor_w = anchor_shape
    ay0 = int(anchor_window["source_y0"])
    ay1 = int(anchor_window["source_y1"])
    ax0 = int(anchor_window["source_x0"])
    ax1 = int(anchor_window["source_x1"])
    win_h = max(1, int(round((ay1 - ay0) * src_h / float(anchor_h))))
    win_w = max(1, int(round((ax1 - ax0) * src_w / float(anchor_w))))
    sy0 = int(round(ay0 * src_h / float(anchor_h)))
    sx0 = int(round(ax0 * src_w / float(anchor_w)))
    sy0 = int(np.clip(sy0, 0, max(0, src_h - win_h)))
    sx0 = int(np.clip(sx0, 0, max(0, src_w - win_w)))
    sy1 = sy0 + win_h
    sx1 = sx0 + win_w
    refine_info: dict[str, int | float | bool | str] = {
        "applied": False,
        "anchor": "swir",
        "anchor_y0": ay0,
        "anchor_x0": ax0,
    }
    if local_refine and refine_with_anchor:
        refined = _refine_source_window_to_anchor(
            source_structure,
            anchor_structure[ay0:ay1, ax0:ax1],
            sy0,
            sy1,
            sx0,
            sx1,
            max_radius_y=max(80, win_h),
            max_radius_x=max(18, win_w // 3),
        )
        sy0, sy1, sx0, sx1, refine_info = refined
        refine_info["anchor"] = "swir"
    return np.asarray(source[sy0:sy1, sx0:sx1, :]), {
        "source_y0": sy0,
        "source_y1": sy1,
        "source_x0": sx0,
        "source_x1": sx1,
        "height": sy1 - sy0,
        "width": sx1 - sx0,
        "local_refine": refine_info,
    }


def _refine_source_window(
    source_structure: np.ndarray,
    rgb_crop: np.ndarray,
    sy0: int,
    sy1: int,
    sx0: int,
    sx1: int,
    max_radius_y: int | None = None,
    max_radius_x: int | None = None,
) -> tuple[int, int, int, int, dict[str, int | float | bool]]:
    win_h = sy1 - sy0
    win_w = sx1 - sx0
    if win_h < 8 or win_w < 8:
        return sy0, sy1, sx0, sx1, {"applied": False, "reason": "window_too_small"}
    rgb_gray = np.mean(normalize_rgb(rgb_crop), axis=2)
    rgb_small = resize_cube(rgb_gray, (win_h, win_w), method="bilinear")
    rgb_feat = 0.15 * _normalize_image(rgb_small) + 0.85 * _gradient(rgb_small)

    max_y = max_radius_y if max_radius_y is not None else max(12, min(180, win_h))
    max_x = max_radius_x if max_radius_x is not None else max(6, min(36, win_w // 2))
    best = (sy0, sx0, -1.0)
    for step, radius_y, radius_x in [(4, max_y, max_x), (1, 8, 5)]:
        center_y, center_x = best[0], best[1]
        y_values = range(center_y - radius_y, center_y + radius_y + 1, step)
        x_values = range(center_x - radius_x, center_x + radius_x + 1, step)
        for y in y_values:
            if y < 0 or y + win_h > source_structure.shape[0]:
                continue
            for x in x_values:
                if x < 0 or x + win_w > source_structure.shape[1]:
                    continue
                src = source_structure[y : y + win_h, x : x + win_w]
                src_feat = 0.15 * _normalize_image(src) + 0.85 * _gradient(src)
                score = _ncc(rgb_feat, src_feat)
                if score > best[2]:
                    best = (int(y), int(x), float(score))
    new_y0, new_x0, score = best
    info = {
        "applied": True,
        "base_y0": sy0,
        "base_x0": sx0,
        "delta_y": int(new_y0 - sy0),
        "delta_x": int(new_x0 - sx0),
        "score": float(score),
    }
    return new_y0, new_y0 + win_h, new_x0, new_x0 + win_w, info


def _refine_source_window_to_anchor(
    source_structure: np.ndarray,
    anchor_structure: np.ndarray,
    sy0: int,
    sy1: int,
    sx0: int,
    sx1: int,
    *,
    max_radius_y: int,
    max_radius_x: int,
) -> tuple[int, int, int, int, dict[str, int | float | bool]]:
    win_h = sy1 - sy0
    win_w = sx1 - sx0
    if win_h < 8 or win_w < 8 or anchor_structure.size == 0:
        return sy0, sy1, sx0, sx1, {"applied": False, "reason": "window_too_small"}
    anchor_intensity = resize_cube(_normalize_image(anchor_structure), (win_h, win_w), method="bilinear")
    anchor_edge = _gradient(anchor_intensity)
    best = (sy0, sx0, -1.0)
    for step, radius_y, radius_x in [(4, max_radius_y, max_radius_x), (1, 10, 6)]:
        center_y, center_x = best[0], best[1]
        for y in range(center_y - radius_y, center_y + radius_y + 1, step):
            if y < 0 or y + win_h > source_structure.shape[0]:
                continue
            for x in range(center_x - radius_x, center_x + radius_x + 1, step):
                if x < 0 or x + win_w > source_structure.shape[1]:
                    continue
                src = _normalize_image(source_structure[y : y + win_h, x : x + win_w])
                src_edge = _gradient(src)
                score = 0.45 * _ncc(anchor_edge, src_edge) + 0.55 * _nmi(anchor_intensity, src)
                if score > best[2]:
                    best = (int(y), int(x), float(score))
    new_y0, new_x0, score = best
    return new_y0, new_y0 + win_h, new_x0, new_x0 + win_w, {
        "applied": True,
        "base_y0": sy0,
        "base_x0": sx0,
        "delta_y": int(new_y0 - sy0),
        "delta_x": int(new_x0 - sx0),
        "score": float(score),
        "metric": "0.45_edge_ncc_0.55_nmi",
    }


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if valid.sum() < 8:
        return -1.0
    aa = aa[valid]
    bb = bb[valid]
    aa = aa - aa.mean()
    bb = bb - bb.mean()
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
    aa = aa[valid]
    bb = bb[valid]
    hist2d, _, _ = np.histogram2d(aa, bb, bins=bins, range=[[0, 1], [0, 1]])
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


def _save_preview(rgb: np.ndarray, path: Path) -> None:
    img = (normalize_rgb(rgb) * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(path)


def _save_registered_sensor_preview(cube: np.ndarray, shape: tuple[int, int], path: Path) -> None:
    img = _normalize_image(np.nanmean(np.asarray(cube, dtype=np.float32), axis=2))
    img = resize_cube(img, shape, method="bilinear")
    Image.fromarray((_normalize_image(img) * 255).round().astype(np.uint8)).save(path)


def _save_overlay(rgb: np.ndarray, cube: np.ndarray, path: Path) -> None:
    rgb_gray = _normalize_image(np.mean(normalize_rgb(rgb), axis=2))
    sensor = _normalize_image(np.nanmean(np.asarray(cube, dtype=np.float32), axis=2))
    sensor = resize_cube(sensor, rgb_gray.shape, method="bilinear")
    overlay = np.stack([rgb_gray, sensor, 0.5 * rgb_gray + 0.5 * sensor], axis=2)
    Image.fromarray((np.clip(overlay, 0, 1) * 255).round().astype(np.uint8)).save(path)


def _save_hsi_overlay(nir_cube: np.ndarray, swir_cube: np.ndarray, path: Path) -> None:
    nir_img = _normalize_image(np.nanmean(np.asarray(nir_cube, dtype=np.float32), axis=2))
    swir_img = _normalize_image(np.nanmean(np.asarray(swir_cube, dtype=np.float32), axis=2))
    nir_img = resize_cube(nir_img, swir_img.shape, method="bilinear")
    overlay = np.stack([nir_img, swir_img, 0.5 * nir_img + 0.5 * swir_img], axis=2)
    Image.fromarray((np.clip(overlay, 0, 1) * 255).round().astype(np.uint8)).save(path)


def _joint_hsi_structure(nir_cube: np.ndarray, swir_cube: np.ndarray) -> np.ndarray:
    nir_img = _normalize_image(np.nanmean(np.asarray(nir_cube, dtype=np.float32), axis=2))
    swir_img = _normalize_image(np.nanmean(np.asarray(swir_cube, dtype=np.float32), axis=2))
    if nir_img.shape != swir_img.shape:
        nir_img = resize_cube(nir_img, swir_img.shape, method="bilinear")
    nir_edge = _gradient(nir_img)
    swir_edge = _gradient(swir_img)
    return _normalize_image(0.2 * nir_img + 0.2 * swir_img + 0.3 * nir_edge + 0.3 * swir_edge)


def _warp_model_is_usable(model: Any, *, min_tie_points: int, min_score: float) -> bool:
    if len(model.tie_points) < min_tie_points:
        return False
    scores = [float(point.score) for point in model.tie_points]
    if not scores or float(np.mean(scores)) < min_score:
        return False
    return True


def _rgb_gray_resized(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    gray = np.mean(normalize_rgb(rgb), axis=2)
    return resize_cube(gray, shape, method="bilinear")


def _save_global_previews(
    rgb_preview: np.ndarray,
    nir_preview: np.ndarray,
    swir_preview: np.ndarray,
    crop: dict[str, int | float],
    preview_scale_y: float,
    preview_scale_x: float,
    preview_dir: Path,
) -> None:
    def box_image(base: np.ndarray) -> np.ndarray:
        img = np.stack([base, base, base], axis=2)
        y0 = int(crop["rgb_y0"] / preview_scale_y)
        y1 = int(crop["rgb_y1"] / preview_scale_y)
        x0 = int(crop["rgb_x0"] / preview_scale_x)
        x1 = int(crop["rgb_x1"] / preview_scale_x)
        img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x0 + 2), :] = [1, 0, 0]
        img[max(0, y0):min(img.shape[0], y1), max(0, x1 - 2):min(img.shape[1], x1), :] = [1, 0, 0]
        img[max(0, y0):min(img.shape[0], y0 + 2), max(0, x0):min(img.shape[1], x1), :] = [1, 0, 0]
        img[max(0, y1 - 2):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1), :] = [1, 0, 0]
        return (img * 255).round().astype(np.uint8)

    Image.fromarray(box_image(rgb_preview)).save(preview_dir / "global_rgb_preview_with_roi.png")
    Image.fromarray(box_image(nir_preview)).save(preview_dir / "global_nir_preview_with_roi.png")
    Image.fromarray(box_image(swir_preview)).save(preview_dir / "global_swir_preview_with_roi.png")


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
    gy = np.zeros_like(image, dtype=np.float32)
    gx = np.zeros_like(image, dtype=np.float32)
    gy[1:-1] = 0.5 * (image[2:] - image[:-2])
    gx[:, 1:-1] = 0.5 * (image[:, 2:] - image[:, :-2])
    return _normalize_image(np.sqrt(gx * gx + gy * gy))


def _valid_brightness(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return ((arr > 0.08) & (arr < 0.95)).astype(np.float32)
