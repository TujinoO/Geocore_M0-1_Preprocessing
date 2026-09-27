from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from geocore_m1_3.config import MaskRefineConfig
from geocore_m1_3.layout.lane_detector import Lane
from geocore_m1_3.mask_refine.components import Component, components_to_mask, connected_components
from geocore_m1_3.utils.geometry import bbox_intersects


@dataclass
class RefinementResult:
    refined_mask: np.ndarray
    kept_components: list[Component]
    removed_components: list[Component]
    warnings: list[dict]
    report: dict


def _ensure_odd(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _binary_filter(mask: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    kernel = _ensure_odd(kernel_size)
    if kernel <= 1:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if operation == "dilate":
        image = image.filter(ImageFilter.MaxFilter(kernel))
    elif operation == "erode":
        image = image.filter(ImageFilter.MinFilter(kernel))
    else:
        raise ValueError(f"Unsupported binary operation: {operation}")
    return np.asarray(image, dtype=np.uint8) > 127


def close_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_filter(_binary_filter(mask, "dilate", kernel_size), "erode", kernel_size)


def open_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    return _binary_filter(_binary_filter(mask, "erode", kernel_size), "dilate", kernel_size)


def fill_small_holes(mask: np.ndarray, max_area: int) -> tuple[np.ndarray, int]:
    if max_area <= 0:
        return mask.astype(bool), 0
    background_components = connected_components(~mask.astype(bool), connectivity=8)
    holes = [component for component in background_components if not component.touches_border and component.area <= max_area]
    if not holes:
        return mask.astype(bool), 0
    filled = mask.astype(bool).copy()
    filled[components_to_mask(holes, mask.shape)] = True
    return filled, len(holes)


def _component_inside_any_lane(component: Component, lanes: list[Lane], margin: int) -> bool:
    x, y = component.centroid
    bbox = component.bbox
    for lane in lanes:
        lx0, ly0, lx1, ly1 = lane.bbox
        lane_bbox = [lx0 - margin, ly0 - margin, lx1 + margin, ly1 + margin]
        if lx0 - margin <= x <= lx1 + margin and ly0 - margin <= y <= ly1 + margin:
            return True
        if bbox_intersects(bbox, lane_bbox):
            return True
    return False


def refine_mask(mask: np.ndarray, lanes: list[Lane], config: MaskRefineConfig) -> RefinementResult:
    original_mask = mask.astype(bool)
    if not config.enable:
        return RefinementResult(
            refined_mask=original_mask,
            kept_components=[],
            removed_components=[],
            warnings=[],
            report={
                "enabled": False,
                "initial_foreground_pixels": int(original_mask.sum()),
                "final_foreground_pixels": int(original_mask.sum()),
                "removed_component_count": 0,
            },
        )

    working = close_mask(original_mask, config.smooth_kernel_size)
    working, filled_hole_count = fill_small_holes(working, config.fill_hole_area_px)
    components = connected_components(working)

    kept: list[Component] = []
    removed: list[Component] = []
    for component in components:
        inside_lane = _component_inside_any_lane(component, lanes, config.lane_margin_px) if lanes else True
        remove_for_size = component.area < config.min_component_area_px and not inside_lane
        remove_for_lane = config.outside_lane_remove and not inside_lane
        if remove_for_size or remove_for_lane:
            removed.append(component)
        else:
            kept.append(component)

    refined = components_to_mask(kept, working.shape)
    removed_area = int(sum(component.area for component in removed))
    initial_area = int(original_mask.sum())
    final_area = int(refined.sum())
    removed_area_ratio = removed_area / max(1, initial_area)
    warnings: list[dict] = []
    if removed_area_ratio > config.removed_area_warning_ratio:
        warnings.append(
            {
                "code": "large_removed_foreground_area",
                "message": "A large foreground area was removed during lane-constrained mask refinement.",
                "removed_area_ratio": removed_area_ratio,
            }
        )

    report = {
        "enabled": True,
        "initial_foreground_pixels": initial_area,
        "post_smooth_foreground_pixels": int(working.sum()),
        "final_foreground_pixels": final_area,
        "filled_hole_count": filled_hole_count,
        "kept_component_count": len(kept),
        "removed_component_count": len(removed),
        "removed_area_pixels": removed_area,
        "removed_area_ratio": removed_area_ratio,
    }
    return RefinementResult(
        refined_mask=refined,
        kept_components=kept,
        removed_components=removed,
        warnings=warnings,
        report=report,
    )
