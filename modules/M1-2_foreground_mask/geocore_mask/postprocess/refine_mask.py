from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from skimage import measure, morphology


def refine_mask(mask: np.ndarray, config: Optional[Dict[str, Any]] = None) -> tuple[np.ndarray, list[str]]:
    cfg = config or {}
    warnings: list[str] = []
    refined = mask.astype(bool)

    min_area = int(cfg.get("min_component_area") or 0)
    if min_area > 0:
        refined = morphology.remove_small_objects(refined, min_size=min_area)

    fill_area = int(cfg.get("fill_hole_area") or 0)
    if fill_area > 0:
        refined = morphology.remove_small_holes(refined, area_threshold=fill_area)

    kernel_size = int(cfg.get("smooth_kernel_size") or 0)
    if kernel_size > 1:
        radius = max(1, kernel_size // 2)
        footprint = morphology.disk(radius)
        refined = morphology.binary_closing(refined, footprint)
        refined = morphology.binary_opening(refined, footprint)

    keep_top_k = cfg.get("keep_top_k")
    if keep_top_k:
        refined = _keep_largest_components(refined, int(keep_top_k))

    area_ratio = float(refined.mean()) if refined.size else 0.0
    if area_ratio < float(cfg.get("min_area_ratio_warning", 0.005)):
        warnings.append("foreground area is unusually small")
    if area_ratio > float(cfg.get("max_area_ratio_warning", 0.85)):
        warnings.append("foreground area is unusually large")

    labels = measure.label(refined)
    component_count = int(labels.max())
    max_components = int(cfg.get("max_component_count_warning", 200))
    if component_count > max_components:
        warnings.append(f"foreground has too many components: {component_count}")

    return refined.astype(np.uint8), warnings


def _keep_largest_components(mask: np.ndarray, keep_top_k: int) -> np.ndarray:
    labels = measure.label(mask)
    props = sorted(measure.regionprops(labels), key=lambda p: p.area, reverse=True)
    keep = {prop.label for prop in props[:keep_top_k]}
    return np.isin(labels, list(keep))
