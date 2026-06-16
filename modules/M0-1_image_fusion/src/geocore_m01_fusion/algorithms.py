"""Fusion algorithms for the M0-1 module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FusionConfig, FusionMode
from .preprocess import (
    box_blur2d,
    edge_weight_from_rgb,
    normalize_mask,
    normalize_reflectance,
    normalize_rgb,
    rgb_detail,
    rgb_luminance,
)
from .resample import downsample_to, resize_cube
from .spectral import StitchResult, calibrate_and_concat, concat_upsampled


@dataclass(slots=True)
class FusionProduct:
    cube: np.ndarray
    wavelengths: list[float]
    band_metadata: list[dict[str, Any]]
    stitch_model: dict[str, Any]
    algorithm_details: dict[str, Any]
    nir_reference: np.ndarray
    swir_reference: np.ndarray
    nir_band_count: int


def fuse_cubes(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    *,
    nir_wavelengths: list[float] | None,
    swir_wavelengths: list[float] | None,
    config: FusionConfig,
    mask: np.ndarray | None = None,
) -> FusionProduct:
    """Dispatch the selected fusion mode."""

    mode = config.normalized_mode()
    if mode == FusionMode.UPSAMPLE_ONLY:
        return upsample_only(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    if mode == FusionMode.FAST_PREVIEW:
        return fast_preview(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    if mode == FusionMode.CLASSICAL:
        return classical(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    if mode == FusionMode.DEEP_UNSUPERVISED:
        return deep_unsupervised(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    raise AssertionError(f"Unhandled fusion mode: {mode}")


def upsample_only(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nir_wavelengths: list[float] | None,
    swir_wavelengths: list[float] | None,
    config: FusionConfig,
    mask: np.ndarray | None = None,
) -> FusionProduct:
    """Baseline: radiometric stitching plus geometric upsampling only."""

    rgb01, nir_ref, swir_ref, target_shape, out_mask = _prepare_inputs(rgb, nir, swir, mask)
    stitched = calibrate_and_concat(nir_ref, swir_ref, nir_wavelengths, swir_wavelengths, mask=out_mask)
    cube = concat_upsampled(stitched, target_shape, config.interpolation)
    cube = _apply_mask(cube, out_mask, config.mask_background_value)
    details = {
        "mode": FusionMode.UPSAMPLE_ONLY.value,
        "interpolation": config.interpolation,
        "description": "Calibrated NIR/SWIR concatenation resized to RGB reference grid.",
    }
    return _product(cube, stitched, details, nir_ref, stitched.swir)


def fast_preview(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nir_wavelengths: list[float] | None,
    swir_wavelengths: list[float] | None,
    config: FusionConfig,
    mask: np.ndarray | None = None,
) -> FusionProduct:
    """Fast RGB-guided detail injection baseline."""

    base = upsample_only(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    rgb01 = normalize_rgb(rgb)
    out_mask = normalize_mask(mask, rgb01.shape[:2])
    cube = _inject_rgb_detail(
        base.cube,
        rgb01,
        strength=config.fast_detail_strength,
        mask=out_mask,
        relative=True,
    )
    cube = _inject_rgb_spatial_detail(
        cube,
        rgb01,
        strength=config.spatial_detail_strength * 0.9,
        mask=out_mask,
        small_radius=config.spatial_detail_small_radius,
        large_radius=config.spatial_detail_large_radius,
    )
    cube = _sensor_back_projection(
        cube,
        base.nir_reference,
        base.swir_reference,
        base.nir_band_count,
        config,
    )
    cube = _apply_mask(cube, out_mask, config.mask_background_value)
    details = {
        "mode": FusionMode.FAST_PREVIEW.value,
        "base_mode": FusionMode.UPSAMPLE_ONLY.value,
        "detail_strength": config.fast_detail_strength,
        "spatial_detail_strength": config.spatial_detail_strength * 0.9,
        "back_projection_iters": config.back_projection_iters,
        "description": "Fast RGB-guided multiplicative and multiscale high-frequency detail injection.",
    }
    return _product(cube, _stitch_from_product(base), details, base.nir_reference, base.swir_reference)


def classical(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nir_wavelengths: list[float] | None,
    swir_wavelengths: list[float] | None,
    config: FusionConfig,
    mask: np.ndarray | None = None,
) -> FusionProduct:
    """Production baseline: low-rank spectral basis plus RGB edge-guided coefficients."""

    base = upsample_only(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    rgb01 = normalize_rgb(rgb)
    out_mask = normalize_mask(mask, rgb01.shape[:2])
    mean, basis, coeff = _fit_low_rank(
        base.cube,
        rank=config.rank,
        max_pixels=config.max_basis_pixels,
        seed=config.random_seed,
    )
    coeff_guided = _inject_rgb_detail(
        coeff,
        rgb01,
        strength=config.classical_detail_strength,
        mask=out_mask,
        relative=False,
    )
    cube = _reconstruct_low_rank(mean, basis, coeff_guided, base.cube.shape)
    # Blend with the conservative baseline so weak bands do not overreact.
    cube = 0.85 * cube + 0.15 * base.cube
    cube = _inject_rgb_spatial_detail(
        cube,
        rgb01,
        strength=config.spatial_detail_strength,
        mask=out_mask,
        small_radius=config.spatial_detail_small_radius,
        large_radius=config.spatial_detail_large_radius,
    )
    cube = _sensor_back_projection(
        cube,
        base.nir_reference,
        base.swir_reference,
        base.nir_band_count,
        config,
    )
    cube = _apply_mask(cube, out_mask, config.mask_background_value)
    details = {
        "mode": FusionMode.CLASSICAL.value,
        "base_mode": FusionMode.UPSAMPLE_ONLY.value,
        "rank": int(basis.shape[0]),
        "detail_strength": config.classical_detail_strength,
        "spatial_detail_strength": config.spatial_detail_strength,
        "spatial_detail_small_radius": config.spatial_detail_small_radius,
        "spatial_detail_large_radius": config.spatial_detail_large_radius,
        "max_basis_pixels": config.max_basis_pixels,
        "back_projection_iters": config.back_projection_iters,
        "description": "Low-rank spectral basis with RGB edge-guided coefficients and multiscale spatial detail preservation.",
    }
    return _product(cube, _stitch_from_product(base), details, base.nir_reference, base.swir_reference)


def deep_unsupervised(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    nir_wavelengths: list[float] | None,
    swir_wavelengths: list[float] | None,
    config: FusionConfig,
    mask: np.ndarray | None = None,
) -> FusionProduct:
    """NumPy model-inspired unsupervised unrolled autoencoder fusion.

    This mode intentionally has no training-label dependency. Each iteration
    applies an encoder bottleneck (low-rank projection), an RGB-guided latent
    update, a decoder reconstruction, and sensor-consistency back-projection.
    It is a runnable research scaffold that can later be replaced by a PyTorch
    autoencoder with the same input/output contract.
    """

    base = upsample_only(rgb, nir, swir, nir_wavelengths, swir_wavelengths, config, mask)
    rgb01 = normalize_rgb(rgb)
    out_mask = normalize_mask(mask, rgb01.shape[:2])
    current = base.cube.copy()
    rank = max(2, min(config.rank, current.shape[2], 24))
    losses: list[dict[str, float]] = []
    for step in range(max(1, config.deep_iterations)):
        mean, basis, coeff = _fit_low_rank(
            current,
            rank=rank,
            max_pixels=config.max_basis_pixels,
            seed=config.random_seed + step,
        )
        strength = config.deep_detail_strength / float(step + 1)
        coeff = _inject_rgb_detail(coeff, rgb01, strength=strength, mask=out_mask, relative=False)
        decoded = _reconstruct_low_rank(mean, basis, coeff, current.shape)
        current = 0.65 * current + 0.35 * decoded
        current = _inject_rgb_spatial_detail(
            current,
            rgb01,
            strength=config.spatial_detail_strength * (0.75 + 0.25 / float(step + 1)),
            mask=out_mask,
            small_radius=config.spatial_detail_small_radius,
            large_radius=config.spatial_detail_large_radius,
        )
        current = _sensor_back_projection(
            current,
            base.nir_reference,
            base.swir_reference,
            base.nir_band_count,
            config,
            iterations=1,
        )
        losses.append(
            {
                "iteration": float(step + 1),
                "nir_consistency_rmse": _rmse(
                    downsample_to(current[:, :, : base.nir_band_count], base.nir_reference.shape[:2]),
                    base.nir_reference,
                ),
                "swir_consistency_rmse": _rmse(
                    downsample_to(current[:, :, base.nir_band_count :], base.swir_reference.shape[:2]),
                    base.swir_reference,
                ),
            }
        )
    current = _apply_mask(current, out_mask, config.mask_background_value)
    details = {
        "mode": FusionMode.DEEP_UNSUPERVISED.value,
        "base_mode": FusionMode.UPSAMPLE_ONLY.value,
        "model_family": "numpy_unrolled_low_rank_autoencoder",
        "rank": int(rank),
        "iterations": int(max(1, config.deep_iterations)),
        "spatial_detail_strength": config.spatial_detail_strength,
        "loss_history": losses,
        "description": "Unsupervised model-inspired low-rank autoencoder scaffold with RGB multiscale detail preservation.",
    }
    return _product(current, _stitch_from_product(base), details, base.nir_reference, base.swir_reference)


def _prepare_inputs(
    rgb: np.ndarray,
    nir: np.ndarray,
    swir: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], np.ndarray | None]:
    rgb01 = normalize_rgb(rgb)
    nir_ref = normalize_reflectance(nir)
    swir_ref = normalize_reflectance(swir)
    if nir_ref.ndim != 3 or swir_ref.ndim != 3:
        raise ValueError("NIR and SWIR inputs must be y,x,band cubes")
    out_mask = normalize_mask(mask, rgb01.shape[:2])
    return rgb01, nir_ref, swir_ref, rgb01.shape[:2], out_mask


def _product(
    cube: np.ndarray,
    stitched: StitchResult,
    algorithm_details: dict[str, Any],
    nir_reference: np.ndarray,
    swir_reference: np.ndarray,
) -> FusionProduct:
    cube = _sanitize_cube(cube)
    return FusionProduct(
        cube=cube.astype(np.float32, copy=False),
        wavelengths=stitched.wavelengths,
        band_metadata=stitched.band_metadata,
        stitch_model=stitched.stitch_model,
        algorithm_details=algorithm_details,
        nir_reference=nir_reference.astype(np.float32, copy=False),
        swir_reference=swir_reference.astype(np.float32, copy=False),
        nir_band_count=nir_reference.shape[2],
    )


def _stitch_from_product(product: FusionProduct) -> StitchResult:
    return StitchResult(
        nir=product.nir_reference,
        swir=product.swir_reference,
        wavelengths=product.wavelengths,
        band_metadata=product.band_metadata,
        stitch_model=product.stitch_model,
    )


def _inject_rgb_detail(
    array: np.ndarray,
    rgb01: np.ndarray,
    *,
    strength: float,
    mask: np.ndarray | None,
    relative: bool,
) -> np.ndarray:
    if strength <= 0:
        return array.copy()
    detail = rgb_detail(rgb01, blur_radius=5)
    edge = edge_weight_from_rgb(rgb01)
    guide = detail * (0.35 + 0.65 * edge)
    if mask is not None:
        guide = guide * mask
    arr = np.asarray(array, dtype=np.float32)
    if relative:
        modifier = 1.0 + strength * guide[:, :, None]
        out = arr * modifier
    else:
        std = np.nanstd(arr.reshape(-1, arr.shape[2]), axis=0).astype(np.float32)
        std[~np.isfinite(std)] = 0.0
        out = arr + strength * guide[:, :, None] * std[None, None, :]
    return _sanitize_cube(out)


def _inject_rgb_spatial_detail(
    cube: np.ndarray,
    rgb01: np.ndarray,
    *,
    strength: float,
    mask: np.ndarray | None,
    small_radius: int,
    large_radius: int,
) -> np.ndarray:
    """Inject RGB multiscale high-frequency detail while preserving spectral shape.

    The low-frequency spectral content remains controlled by the upsampled
    NIR/SWIR cube.  RGB contributes only normalized high-frequency residuals,
    weighted by RGB edges, so cracks, grain boundaries, and core-box borders are
    restored on the RGB grid without replacing the hyperspectral spectrum.
    """

    if strength <= 0:
        return np.asarray(cube, dtype=np.float32).copy()
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("spatial detail injection expects y,x,band cube")
    guide = _rgb_multiscale_detail_guide(
        rgb01,
        small_radius=small_radius,
        large_radius=large_radius,
    )
    if mask is not None:
        guide = guide * mask.astype(np.float32)

    flat = arr.reshape(-1, arr.shape[2])
    band_std = np.nanstd(flat, axis=0).astype(np.float32)
    band_mean = np.nanmean(flat, axis=0).astype(np.float32)
    band_std[~np.isfinite(band_std)] = 0.0
    band_mean[~np.isfinite(band_mean)] = 0.0
    # Use both multiplicative and additive terms: the multiplicative term keeps
    # spectral ratios stable, while a small additive term restores detail in
    # low-reflectance bands where pure ratio modulation is visually too weak.
    ratio = np.clip(1.0 + strength * guide, 0.55, 1.65).astype(np.float32)
    additive_scale = np.maximum(0.20 * band_std, 0.03 * np.maximum(band_mean, 1e-4))
    out = arr * ratio[:, :, None] + (0.35 * strength) * guide[:, :, None] * additive_scale[None, None, :]
    return _sanitize_cube(out)


def _rgb_multiscale_detail_guide(
    rgb01: np.ndarray,
    *,
    small_radius: int,
    large_radius: int,
) -> np.ndarray:
    lum = rgb_luminance(rgb01)
    small_radius = max(1, int(small_radius))
    large_radius = max(small_radius + 1, int(large_radius))
    blur_small = box_blur2d(lum, radius=small_radius)
    blur_large = box_blur2d(lum, radius=large_radius)
    fine = lum - blur_small
    medium = blur_small - blur_large
    detail = fine + 0.55 * medium
    finite = detail[np.isfinite(detail)]
    if finite.size == 0:
        return np.zeros(lum.shape, dtype=np.float32)
    scale = float(np.percentile(np.abs(finite), 98))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros(lum.shape, dtype=np.float32)
    detail = np.clip(detail / scale, -1.0, 1.0)
    edge = edge_weight_from_rgb(rgb01)
    guide = detail * (0.35 + 0.65 * edge)
    return np.clip(guide, -1.0, 1.0).astype(np.float32)


def _fit_low_rank(
    cube: np.ndarray,
    *,
    rank: int,
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(cube, dtype=np.float32)
    h, w, b = arr.shape
    flat = arr.reshape(-1, b)
    valid = np.isfinite(flat).all(axis=1)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        mean = np.zeros((b,), dtype=np.float32)
        basis = np.eye(min(rank, b), b, dtype=np.float32)
        coeff = np.zeros((h, w, basis.shape[0]), dtype=np.float32)
        return mean, basis, coeff
    if valid_idx.size > max_pixels:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(valid_idx, size=max_pixels, replace=False)
    else:
        sample_idx = valid_idx
    sample = flat[sample_idx]
    mean = np.nanmean(sample, axis=0).astype(np.float32)
    centered = sample - mean
    k = max(1, min(int(rank), b, centered.shape[0]))
    basis = _stable_pca_basis(centered, k, seed=seed)
    k = int(basis.shape[0])
    coeff_flat = _project_onto_basis(np.nan_to_num(flat, nan=0.0) - mean, basis)
    return mean, basis, coeff_flat.reshape(h, w, k).astype(np.float32)


def _reconstruct_low_rank(
    mean: np.ndarray,
    basis: np.ndarray,
    coeff: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    h, w, _ = shape
    coeff_flat = np.asarray(coeff, dtype=np.float32).reshape(-1, coeff.shape[2])
    flat = np.broadcast_to(mean.astype(np.float32), (coeff_flat.shape[0], mean.shape[0])).copy()
    for i in range(basis.shape[0]):
        flat += coeff_flat[:, i : i + 1] * basis[i : i + 1, :]
    return flat.reshape(h, w, mean.shape[0]).astype(np.float32)


def _stable_pca_basis(centered: np.ndarray, rank: int, *, seed: int) -> np.ndarray:
    """Fit a small PCA basis without relying on LAPACK-backed SVD."""

    arr = np.nan_to_num(np.asarray(centered, dtype=np.float32))
    samples, bands = arr.shape
    k = max(1, min(int(rank), bands, samples))
    if samples == 0 or bands == 0:
        return np.eye(k, bands, dtype=np.float32)
    cov = np.einsum("nb,nc->bc", arr, arr, optimize=False).astype(np.float32)
    cov /= float(max(samples - 1, 1))
    cov = 0.5 * (cov + cov.T)
    cov_work = cov.copy()
    rng = np.random.default_rng(seed)
    basis: list[np.ndarray] = []
    for _ in range(k):
        diag = np.diag(cov_work)
        if not np.isfinite(diag).any() or float(np.nanmax(np.abs(cov_work))) <= 1e-10:
            break
        v = np.zeros((bands,), dtype=np.float32)
        v[int(np.nanargmax(diag))] = 1.0
        v += (0.01 * rng.standard_normal(bands)).astype(np.float32)
        v = _normalize_vector(_orthogonalize(v, basis))
        if v is None:
            break
        for _step in range(80):
            candidate = _cov_matvec(cov_work, v)
            candidate = _normalize_vector(_orthogonalize(candidate, basis))
            if candidate is None:
                break
            if abs(float(np.sum(candidate * v))) > 0.9999:
                v = candidate
                break
            v = candidate
        eigenvalue = float(np.sum(v * _cov_matvec(cov_work, v)))
        if not np.isfinite(eigenvalue) or eigenvalue <= 1e-10:
            break
        basis.append(v.astype(np.float32))
        cov_work -= eigenvalue * (v[:, None] * v[None, :])
        cov_work = 0.5 * (cov_work + cov_work.T)
    while len(basis) < k:
        v = np.zeros((bands,), dtype=np.float32)
        v[len(basis) % bands] = 1.0
        v = _normalize_vector(_orthogonalize(v, basis))
        if v is None:
            break
        basis.append(v.astype(np.float32))
    if not basis:
        return np.eye(k, bands, dtype=np.float32)
    return np.stack(basis[:k], axis=0).astype(np.float32)


def _project_onto_basis(array: np.ndarray, basis: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    out = np.empty((arr.shape[0], basis.shape[0]), dtype=np.float32)
    for i in range(basis.shape[0]):
        out[:, i] = np.sum(arr * basis[i][None, :], axis=1)
    return out


def _cov_matvec(cov: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.sum(cov * vector[None, :], axis=1).astype(np.float32)


def _orthogonalize(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float32).copy()
    for component in basis:
        out -= float(np.sum(out * component)) * component
    return out


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.sqrt(np.sum(vector * vector)))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def _sensor_back_projection(
    fused: np.ndarray,
    nir_ref: np.ndarray,
    swir_ref: np.ndarray,
    nir_band_count: int,
    config: FusionConfig,
    *,
    iterations: int | None = None,
) -> np.ndarray:
    out = np.asarray(fused, dtype=np.float32).copy()
    iters = config.back_projection_iters if iterations is None else iterations
    for _ in range(max(0, iters)):
        down_nir = downsample_to(out[:, :, :nir_band_count], nir_ref.shape[:2])
        nir_residual = nir_ref - down_nir
        out[:, :, :nir_band_count] += config.back_projection_weight * resize_cube(
            nir_residual, out.shape[:2], method="bilinear"
        )
        down_swir = downsample_to(out[:, :, nir_band_count:], swir_ref.shape[:2])
        swir_residual = swir_ref - down_swir
        out[:, :, nir_band_count:] += config.back_projection_weight * resize_cube(
            swir_residual, out.shape[:2], method="bilinear"
        )
    return _sanitize_cube(out)


def _apply_mask(cube: np.ndarray, mask: np.ndarray | None, value: float) -> np.ndarray:
    if mask is None:
        return cube
    out = cube.copy()
    out[mask <= 0.5] = value
    return out


def _sanitize_cube(cube: np.ndarray) -> np.ndarray:
    out = np.asarray(cube, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=1.5, neginf=-0.2)
    return np.clip(out, -0.2, 1.5).astype(np.float32, copy=False)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.nanmean(diff * diff)))
