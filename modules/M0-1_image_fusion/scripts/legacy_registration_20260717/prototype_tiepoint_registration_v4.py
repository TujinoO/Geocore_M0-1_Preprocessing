"""Prototype guarded grid-tie-point registration for GeoCoreFusion v4.

This script reads the current GeoCoreFusion registration result in memory,
estimates one shared residual warp from the joint NIR/SWIR structure to RGB,
and writes diagnostics only. It does not modify either project or any cube.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


GEOCOREFUSION_ROOT = Path(r"E:\Code\GeoCoreFusion")
PREPROCESSING_M01_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(GEOCOREFUSION_ROOT / "src"))
sys.path.insert(0, str(PREPROCESSING_M01_SRC))

from geocorefusion.config import load_config  # noqa: E402
from geocorefusion.dataset import discover_triplet, normalize_image  # noqa: E402
from geocorefusion.registration import (  # noqa: E402
    _corr,
    _modality_feature,
    estimate_registration,
    estimate_roi_registration,
)
from geocorefusion.pipeline import _analysis_shape  # noqa: E402
from geocorefusion.roi import choose_roi  # noqa: E402
from geocore_m01_fusion.registration import (  # noqa: E402
    interpolate_displacement_field as original_interpolate_displacement_field,
    register_local_warp as original_register_local_warp,
)


@dataclass(slots=True)
class TiePoint:
    ref_y: float
    ref_x: float
    moving_y: float
    moving_x: float
    shift_y: float
    shift_x: float
    score: float
    margin: float
    backward_error: float


def _subpixel_peak(response: np.ndarray, py: int, px: int) -> tuple[float, float]:
    dy = dx = 0.0
    if 0 < px < response.shape[1] - 1:
        left, center, right = map(float, response[py, px - 1 : px + 2])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-6:
            dx = float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))
    if 0 < py < response.shape[0] - 1:
        top, center, bottom = map(float, response[py - 1 : py + 2, px])
        denominator = top - 2.0 * center + bottom
        if abs(denominator) > 1e-6:
            dy = float(np.clip(0.5 * (top - bottom) / denominator, -0.75, 0.75))
    return dy, dx


def _match_one(
    reference: np.ndarray,
    moving: np.ndarray,
    y: int,
    x: int,
    template_radius: int,
    search_radius: int,
) -> tuple[float, float, float, float] | None:
    r = template_radius
    s = search_radius
    if y - r < 0 or y + r >= reference.shape[0] or x - r < 0 or x + r >= reference.shape[1]:
        return None
    template = reference[y - r : y + r + 1, x - r : x + r + 1]
    if float(np.std(template)) < 0.035:
        return None
    y0, y1 = max(0, y - r - s), min(moving.shape[0], y + r + s + 1)
    x0, x1 = max(0, x - r - s), min(moving.shape[1], x + r + s + 1)
    search = moving[y0:y1, x0:x1]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    px, py = location
    suppressed = response.copy()
    suppressed[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2] = -1.0
    second = float(np.max(suppressed)) if suppressed.size > 9 else -1.0
    sub_y, sub_x = _subpixel_peak(response, py, px)
    moving_y = y0 + py + sub_y + r
    moving_x = x0 + px + sub_x + r
    return float(moving_y), float(moving_x), float(score), float(score - second)


def estimate_tie_points(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    grid_rows: int = 13,
    grid_cols: int = 8,
    template_radius: int = 7,
    search_radius: int = 8,
) -> tuple[list[TiePoint], list[dict[str, float | str]]]:
    height, width = reference.shape
    ys = np.linspace(template_radius + 2, height - template_radius - 3, grid_rows).round().astype(int)
    xs = np.linspace(template_radius + 2, width - template_radius - 3, grid_cols).round().astype(int)
    accepted: list[TiePoint] = []
    rejected: list[dict[str, float | str]] = []
    for y in ys:
        for x in xs:
            forward = _match_one(reference, moving, int(y), int(x), template_radius, search_radius)
            if forward is None:
                rejected.append({"ref_y": float(y), "ref_x": float(x), "reason": "low_texture_or_border"})
                continue
            moving_y, moving_x, score, margin = forward
            backward = _match_one(
                moving,
                reference,
                int(round(moving_y)),
                int(round(moving_x)),
                template_radius,
                search_radius,
            )
            if backward is None:
                rejected.append({"ref_y": float(y), "ref_x": float(x), "score": score, "reason": "backward_failed"})
                continue
            back_y, back_x, _, _ = backward
            backward_error = float(np.hypot(back_y - y, back_x - x))
            point = TiePoint(
                ref_y=float(y),
                ref_x=float(x),
                moving_y=moving_y,
                moving_x=moving_x,
                shift_y=moving_y - float(y),
                shift_x=moving_x - float(x),
                score=score,
                margin=margin,
                backward_error=backward_error,
            )
            if score < 0.27 or margin < 0.012 or backward_error > 1.75:
                payload = asdict(point)
                payload["reason"] = "confidence_guard"
                rejected.append(payload)
            else:
                accepted.append(point)

    if len(accepted) < 8:
        return [], rejected
    shifts = np.asarray([[p.shift_y, p.shift_x] for p in accepted], dtype=np.float32)
    median = np.median(shifts, axis=0)
    mad = np.median(np.abs(shifts - median), axis=0) + 0.35
    kept: list[TiePoint] = []
    for point, shift in zip(accepted, shifts, strict=True):
        robust_z = np.max(np.abs(shift - median) / (1.4826 * mad))
        if robust_z > 3.5 or float(np.hypot(*shift)) > search_radius * 1.15:
            payload = asdict(point)
            payload["reason"] = "robust_displacement_guard"
            rejected.append(payload)
        else:
            kept.append(point)
    return kept, rejected


def estimate_original_tie_points(
    reference: np.ndarray,
    moving: np.ndarray,
) -> tuple[list[TiePoint], list[dict[str, float | str]], np.ndarray, np.ndarray]:
    model = original_register_local_warp(
        reference,
        moving,
        approx_scale_y=1.0,
        approx_scale_x=1.0,
        grid_rows=13,
        grid_cols=8,
        template_radius=9,
        search_radius_y=5,
        search_radius_x=5,
        min_tie_points=16,
    )
    points = [
        TiePoint(
            ref_y=float(point.ref_y),
            ref_x=float(point.ref_x),
            moving_y=float(point.moving_y),
            moving_x=float(point.moving_x),
            shift_y=float(point.delta_y),
            shift_x=float(point.delta_x),
            score=float(point.score),
            margin=0.0,
            backward_error=0.0,
        )
        for point in model.tie_points
    ]
    rejected = [dict(point.to_dict(), reason="original_score_or_mad_guard") for point in model.rejected_tie_points]
    yy, xx = np.indices(reference.shape, dtype=np.float32)
    shift_y, shift_x = original_interpolate_displacement_field(model, yy, xx)
    shift_y = cv2.GaussianBlur(shift_y.astype(np.float32), (0, 0), 1.6)
    shift_x = cv2.GaussianBlur(shift_x.astype(np.float32), (0, 0), 1.6)
    return points, rejected, shift_y, shift_x


def estimate_field(
    reference: np.ndarray,
    moving: np.ndarray,
    matcher: str,
) -> tuple[list[TiePoint], list[dict[str, float | str]], np.ndarray, np.ndarray]:
    if matcher == "original":
        return estimate_original_tie_points(reference, moving)
    reference_feature = _modality_feature(reference)
    moving_feature = _modality_feature(moving)
    points, rejected = estimate_tie_points(reference_feature, moving_feature)
    if len(points) < 8:
        return points, rejected, np.zeros(reference.shape, dtype=np.float32), np.zeros(reference.shape, dtype=np.float32)
    shift_y, shift_x = idw_field(points, reference.shape)
    return points, rejected, shift_y, shift_x


def idw_field(points: list[TiePoint], shape: tuple[int, int], neighbours: int = 10) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    coordinates = np.asarray([[p.ref_y, p.ref_x] for p in points], dtype=np.float32)
    shifts = np.asarray([[p.shift_y, p.shift_x] for p in points], dtype=np.float32)
    confidence = np.asarray([max(0.05, p.score - 0.20) ** 2 for p in points], dtype=np.float32)
    yy, xx = np.indices(shape, dtype=np.float32)
    query = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    tree = cKDTree(coordinates)
    distances, indices = tree.query(query, k=min(neighbours, len(points)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = confidence[indices] / (distances * distances + 18.0)
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
    dense = np.sum(weights[:, :, None] * shifts[indices], axis=1).reshape(height, width, 2)
    dense[:, :, 0] = cv2.GaussianBlur(dense[:, :, 0], (0, 0), 2.2)
    dense[:, :, 1] = cv2.GaussianBlur(dense[:, :, 1], (0, 0), 2.2)
    return dense[:, :, 0].astype(np.float32), dense[:, :, 1].astype(np.float32)


def warp(image: np.ndarray, shift_y: np.ndarray, shift_x: np.ndarray, factor: float) -> np.ndarray:
    yy, xx = np.indices(image.shape, dtype=np.float32)
    return cv2.remap(
        np.asarray(image, dtype=np.float32),
        xx + float(factor) * shift_x,
        yy + float(factor) * shift_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _jacobian_min(shift_y: np.ndarray, shift_x: np.ndarray, factor: float) -> float:
    dy_y, dy_x = np.gradient(float(factor) * shift_y)
    dx_y, dx_x = np.gradient(float(factor) * shift_x)
    determinant = (1.0 + dx_x) * (1.0 + dy_y) - dx_y * dy_x
    return float(np.percentile(determinant, 1.0))


def _overlay(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref = normalize_image(_modality_feature(reference))
    mov = normalize_image(_modality_feature(moving))
    return np.stack([mov, ref, 0.5 * (mov + ref)], axis=2)


def registration_scores(reference: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> dict[str, float]:
    return {
        "nir_rgb": _corr(_modality_feature(reference), _modality_feature(nir)),
        "swir_rgb": _corr(_modality_feature(reference), _modality_feature(swir)),
        "nir_swir": _corr(_modality_feature(nir), _modality_feature(swir)),
    }


def select_warp_factor(
    reference: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    shift_y: np.ndarray,
    shift_x: np.ndarray,
    *,
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    baseline = registration_scores(reference, nir, swir)
    trials: list[dict[str, float | bool]] = []
    selected: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for factor in (1.0, 0.75, 0.5, 0.25, 0.0):
        if target == "swir":
            nir_trial = nir
            swir_trial = warp(swir, shift_y, shift_x, factor)
        elif target == "nir":
            nir_trial = warp(nir, shift_y, shift_x, factor)
            swir_trial = swir
        elif target == "shared":
            nir_trial = warp(nir, shift_y, shift_x, factor)
            swir_trial = warp(swir, shift_y, shift_x, factor)
        else:
            raise ValueError(target)
        scores = registration_scores(reference, nir_trial, swir_trial)
        jacobian = _jacobian_min(shift_y, shift_x, factor)
        if target == "swir":
            feasible = bool(scores["nir_swir"] >= baseline["nir_swir"] - 0.020 and jacobian > 0.50)
            objective = (scores["swir_rgb"] - baseline["swir_rgb"]) + 0.40 * (scores["nir_swir"] - baseline["nir_swir"])
        elif target == "nir":
            feasible = bool(scores["nir_swir"] >= baseline["nir_swir"] - 0.020 and jacobian > 0.50)
            objective = (scores["nir_rgb"] - baseline["nir_rgb"]) + 0.40 * (scores["nir_swir"] - baseline["nir_swir"])
        else:
            feasible = bool(scores["nir_swir"] >= baseline["nir_swir"] - 0.012 and jacobian > 0.45)
            objective = (scores["nir_rgb"] - baseline["nir_rgb"]) + (scores["swir_rgb"] - baseline["swir_rgb"]) + 0.25 * (scores["nir_swir"] - baseline["nir_swir"])
        trials.append({"factor": factor, "feasible": feasible, "objective": objective, "jacobian_p01": jacobian, **scores})
        if feasible and (selected is None or objective > selected[0]):
            selected = (objective, factor, nir_trial, swir_trial)
    assert selected is not None
    _, factor, selected_nir, selected_swir = selected
    selected_trial = next(item for item in trials if item["factor"] == factor)
    return selected_nir, selected_swir, {
        "target": target,
        "baseline": baseline,
        "trials": trials,
        "selected_factor": factor,
        "selected": selected_trial,
    }


def _save_u8(path: Path, image: np.ndarray) -> None:
    normalized = np.clip(image, 0.0, 1.0)
    Image.fromarray((normalized * 255.0).round().astype(np.uint8)).save(path)


def _tiepoint_preview(reference: np.ndarray, moving: np.ndarray, points: list[TiePoint], path: Path) -> None:
    left = (normalize_image(reference) * 255.0).round().astype(np.uint8)
    right = (normalize_image(moving) * 255.0).round().astype(np.uint8)
    canvas = Image.fromarray(np.concatenate([left, right], axis=1)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    offset = reference.shape[1]
    for point in points:
        color = (40, int(np.clip(point.score, 0, 1) * 255), 255)
        draw.line((point.ref_x, point.ref_y, offset + point.moving_x, point.moving_y), fill=color, width=1)
        draw.ellipse((point.ref_x - 2, point.ref_y - 2, point.ref_x + 2, point.ref_y + 2), outline=(255, 220, 0))
        draw.ellipse((offset + point.moving_x - 2, point.moving_y - 2, offset + point.moving_x + 2, point.moving_y + 2), outline=(0, 255, 180))
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--matcher", choices=("original", "enhanced"), default="original")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cv2.setNumThreads(1)
    cv2.setRNGSeed(20260717)
    cv2.setUseOptimized(False)
    config = load_config(args.config)
    dataset = discover_triplet(config.data_dir)
    coarse = estimate_registration(dataset, config.registration)
    roi = choose_roi(config.roi, coarse, dataset.rgb.meta.shape[:2])
    shape = _analysis_shape(roi, coarse)
    refined = estimate_roi_registration(dataset, coarse, roi, shape, config.registration)

    reference = refined.reference_structure
    nir = refined.nir_aligned
    swir = refined.swir_aligned
    initial_scores = registration_scores(reference, nir, swir)
    stages: list[dict[str, object]] = []
    all_points: list[TiePoint] = []
    all_rejected: list[dict[str, float | str]] = []

    swir_points, swir_rejected, swir_dy, swir_dx = estimate_field(reference, swir, args.matcher)
    if len(swir_points) >= 8:
        _tiepoint_preview(reference, swir, swir_points, args.output / "tie_points_swir_to_rgb.png")
        nir, swir, swir_stage = select_warp_factor(reference, nir, swir, swir_dy, swir_dx, target="swir")
        swir_stage["source"] = "swir_to_rgb_tie_points"
        swir_stage["tie_point_count"] = len(swir_points)
        swir_stage["rejected_tie_point_count"] = len(swir_rejected)
        stages.append(swir_stage)
        all_points.extend(swir_points)
        all_rejected.extend(swir_rejected)

    nir_points, nir_rejected, nir_dy, nir_dx = estimate_field(reference, nir, args.matcher)
    if len(nir_points) >= 8:
        _tiepoint_preview(reference, nir, nir_points, args.output / "tie_points_nir_to_rgb.png")
        nir, swir, nir_stage = select_warp_factor(reference, nir, swir, nir_dy, nir_dx, target="nir")
        nir_stage["source"] = "nir_to_rgb_tie_points"
        nir_stage["tie_point_count"] = len(nir_points)
        nir_stage["rejected_tie_point_count"] = len(nir_rejected)
        stages.append(nir_stage)
        all_points.extend(nir_points)
        all_rejected.extend(nir_rejected)

    joint = normalize_image(0.5 * normalize_image(nir) + 0.5 * normalize_image(swir))
    ref_feature = _modality_feature(reference)
    joint_feature = _modality_feature(joint)
    points, rejected, shift_y, shift_x = estimate_field(reference, joint, args.matcher)
    if len(points) < 8:
        raise RuntimeError(f"Only {len(points)} guarded tie points survived")
    _tiepoint_preview(reference, joint, points, args.output / "tie_points_joint_to_rgb.png")
    nir_final, swir_final, joint_stage = select_warp_factor(reference, nir, swir, shift_y, shift_x, target="shared")
    joint_stage["tie_point_count"] = len(points)
    joint_stage["rejected_tie_point_count"] = len(rejected)
    stages.append(joint_stage)
    all_points.extend(points)
    all_rejected.extend(rejected)
    factor = float(joint_stage["selected_factor"])

    _save_u8(args.output / "before_joint_rgb_overlay.png", _overlay(reference, normalize_image(0.5 * normalize_image(refined.nir_aligned) + 0.5 * normalize_image(refined.swir_aligned))))
    joint_final = normalize_image(0.5 * normalize_image(nir_final) + 0.5 * normalize_image(swir_final))
    _save_u8(args.output / "after_joint_rgb_overlay.png", _overlay(reference, joint_final))
    _save_u8(args.output / "after_nir_rgb_overlay.png", _overlay(reference, nir_final))
    _save_u8(args.output / "after_swir_rgb_overlay.png", _overlay(reference, swir_final))
    _tiepoint_preview(ref_feature, joint_feature, points, args.output / "tie_points.png")
    displacement = np.sqrt(shift_x * shift_x + shift_y * shift_y)
    _save_u8(args.output / "displacement_magnitude.png", normalize_image(displacement))
    report = {
        "config": str(args.config),
        "matcher": args.matcher,
        "roi": roi,
        "analysis_shape": list(shape),
        "tie_point_count": len(all_points),
        "rejected_tie_point_count": len(all_rejected),
        "tie_points": [asdict(point) for point in all_points],
        "initial_scores": initial_scores,
        "stages": stages,
        "selected_factor": factor,
        "selected": joint_stage["selected"],
        "final_scores": registration_scores(reference, nir_final, swir_final),
        "displacement": {
            "median": float(np.median(displacement)),
            "p95": float(np.percentile(displacement, 95)),
            "max": float(np.max(displacement)),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tie_points": len(points), "selected": report["selected"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
