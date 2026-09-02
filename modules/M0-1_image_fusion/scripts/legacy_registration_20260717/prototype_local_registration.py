from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT = Path(r"E:\Code\GeoCoreFusion")
MODULE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from geocorefusion.config import load_config  # noqa: E402
from geocorefusion.dataset import discover_triplet, normalize_image, rgb_structure  # noqa: E402
from geocorefusion.pipeline import _analysis_shape  # noqa: E402
from geocorefusion.registration import analysis_rgb_grid, estimate_registration, sample_cube_on_rgb_grid  # noqa: E402


def feature(image: np.ndarray) -> np.ndarray:
    base = (normalize_image(image) * 255).round().astype(np.uint8)
    base = cv2.createCLAHE(2.0, (8, 8)).apply(base).astype(np.float32) / 255.0
    gx = cv2.Scharr(base, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(base, cv2.CV_32F, 0, 1)
    grad = normalize_image(cv2.magnitude(gx, gy))
    threshold = float(np.percentile(grad, 72))
    binary = (grad >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    proximity = np.exp(-distance / 2.5).astype(np.float32)
    return normalize_image(0.70 * proximity + 0.30 * cv2.GaussianBlur(grad, (0, 0), 0.8))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 0 else float("nan")


def overlay(path: Path, a: np.ndarray, b: np.ndarray) -> None:
    aa, bb = normalize_image(a), normalize_image(b)
    rgb = np.stack([aa, bb, 0.5 * (aa + bb)], axis=2)
    rgb = cv2.resize(rgb, (rgb.shape[1] * 4, rgb.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(path)


def estimate(ref: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    ref_f = feature(ref)
    mov_f = feature(mov)
    shift, response = cv2.phaseCorrelate(ref_f, mov_f)
    starts = [
        np.array([[1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[1, 0, shift[0]], [0, 1, shift[1]]], np.float32),
    ]
    candidates: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for motion in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
        for start in starts:
            warp = start.copy()
            try:
                ecc, warp = cv2.findTransformECC(
                    ref_f,
                    mov_f,
                    warp,
                    motion,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 800, 1e-7),
                    None,
                    5,
                )
            except cv2.error:
                continue
            aligned = cv2.warpAffine(mov_f, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101)
            score = corr(ref_f, aligned)
            candidates.append((score, float(ecc), warp.copy(), aligned))
    score, ecc, warp, aligned = max(candidates, key=lambda v: v[0])
    return warp, aligned, {"phase_dx": shift[0], "phase_dy": shift[1], "phase_response": response, "ecc": ecc, "corr": score}


def strip_refine(ref: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    h, w = ref.shape
    centers = np.linspace(max(30, h * 0.08), min(h - 31, h * 0.92), 9)
    half = max(28, int(round(h / 8)))
    records = []
    accepted_y, accepted_dx, accepted_dy = [], [], []
    for center in centers:
        y0 = max(0, int(round(center)) - half)
        y1 = min(h, int(round(center)) + half + 1)
        ref_patch = ref[y0:y1]
        best = (corr(ref_patch, moving[y0:y1]), 0, 0)
        second = -1.0
        yy, xx = np.indices((y1 - y0, w), dtype=np.float32)
        yy += y0
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                sampled = cv2.remap(moving, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                score = corr(ref_patch[:, 8:-8], sampled[:, 8:-8])
                if score > best[0]:
                    second = best[0]
                    best = (score, dx, dy)
                elif score > second:
                    second = score
        gain = best[0] - corr(ref_patch[:, 8:-8], moving[y0:y1, 8:-8])
        peak_margin = best[0] - second
        accepted = bool(gain > 0.004 and peak_margin > 0.0002)
        records.append({"y": float(center), "dx": int(best[1]), "dy": int(best[2]), "score": float(best[0]), "gain": float(gain), "accepted": accepted})
        if accepted:
            accepted_y.append(float(center))
            accepted_dx.append(float(best[1]))
            accepted_dy.append(float(best[2]))
    if len(accepted_y) < 3:
        return moving, {"records": records, "applied": False}
    rows = np.arange(h, dtype=np.float32)
    dx_rows = np.interp(rows, accepted_y, accepted_dx).astype(np.float32)
    dy_rows = np.interp(rows, accepted_y, accepted_dy).astype(np.float32)
    kernel = min(61, h // 2 * 2 - 1)
    kernel = max(5, kernel)
    dx_rows = cv2.GaussianBlur(dx_rows[:, None], (1, kernel), 0).reshape(-1)
    dy_rows = cv2.GaussianBlur(dy_rows[:, None], (1, kernel), 0).reshape(-1)
    yy, xx = np.indices((h, w), dtype=np.float32)
    refined = cv2.remap(moving, xx + dx_rows[:, None], yy + dy_rows[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return refined, {
        "records": records,
        "applied": True,
        "dx_min": float(dx_rows.min()),
        "dx_max": float(dx_rows.max()),
        "dy_min": float(dy_rows.min()),
        "dy_max": float(dy_rows.max()),
    }


def run(config_path: Path, out: Path) -> None:
    config = load_config(config_path)
    dataset = discover_triplet(config.data_dir)
    registration = estimate_registration(dataset, config.registration)
    roi = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))["rgb_roi"]
    shape = _analysis_shape(roi, registration)
    yy, xx = analysis_rgb_grid(roi, *shape)
    rgb_crop = np.asarray(dataset.rgb.cube[roi["y"]:roi["y"] + roi["height"], roi["x"]:roi["x"] + roi["width"], :3])
    ref = rgb_structure(rgb_crop, target_shape=shape)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    aligned_outputs = {}
    for name, sensor, model, waves in (
        ("nir", dataset.nir, registration.nir, [850, 1050, 1250, 1400]),
        ("swir", dataset.swir, registration.swir, [1050, 1250, 1650, 2200]),
    ):
        ids = [int(np.argmin(np.abs(sensor.meta.wavelengths - w))) for w in waves]
        cube = sample_cube_on_rgb_grid(sensor.cube, model, yy, xx, bands=ids)
        mov = normalize_image(np.nanmean(cube, axis=2))
        warp, aligned_feature, metrics = estimate(ref, mov)
        aligned = cv2.warpAffine(mov, warp, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101)
        refined_feature, strip_details = strip_refine(feature(ref), aligned_feature)
        overlay(out / f"{name}_before.png", feature(ref), feature(mov))
        overlay(out / f"{name}_after.png", feature(ref), aligned_feature)
        overlay(out / f"{name}_strip_refined.png", feature(ref), refined_feature)
        metrics["before_corr"] = corr(feature(ref), feature(mov))
        metrics["strip_refined_corr"] = corr(feature(ref), refined_feature)
        metrics["strip_refinement"] = strip_details
        metrics["warp"] = warp.tolist()
        report[name] = metrics
        overlap_ids = [int(np.argmin(np.abs(sensor.meta.wavelengths - w))) for w in [1050, 1200, 1350, 1450]]
        overlap_cube = sample_cube_on_rgb_grid(sensor.cube, model, yy, xx, bands=overlap_ids)
        overlap_image = normalize_image(np.nanmean(overlap_cube, axis=2))
        overlap_aligned = cv2.warpAffine(overlap_image, warp, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101)
        aligned_outputs[name] = {"warp": warp, "overlap": overlap_aligned}

    nir_to_swir, nir_overlap_joint, joint_metrics = estimate(aligned_outputs["swir"]["overlap"], aligned_outputs["nir"]["overlap"])
    nir_total = np.vstack([aligned_outputs["nir"]["warp"], [0, 0, 1]]) @ np.vstack([nir_to_swir, [0, 0, 1]])
    nir_original_ids = [int(np.argmin(np.abs(dataset.nir.meta.wavelengths - w))) for w in [850, 1050, 1250, 1400]]
    nir_original = normalize_image(np.nanmean(sample_cube_on_rgb_grid(dataset.nir.cube, registration.nir, yy, xx, bands=nir_original_ids), axis=2))
    nir_joint_rgb = cv2.warpAffine(nir_original, nir_total[:2].astype(np.float32), (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT101)
    overlay(out / "nir_joint_swir_anchor.png", feature(ref), feature(nir_joint_rgb))
    overlay(out / "nir_swir_overlap_joint.png", feature(aligned_outputs["swir"]["overlap"]), feature(nir_overlap_joint))
    joint_metrics["rgb_corr_after_joint"] = corr(feature(ref), feature(nir_joint_rgb))
    joint_metrics["nir_total_warp"] = nir_total[:2].tolist()
    report["nir_to_swir_joint"] = joint_metrics
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    root = MODULE_ROOT / "runs" / "legacy_registration" / "local_registration_prototype"
    run(PROJECT / "configs" / "3dssz_roi.yaml", root / "3dssz")
    run(PROJECT / "configs" / "zkh3_roi.yaml", root / "zkh3")
