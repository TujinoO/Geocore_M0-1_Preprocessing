from __future__ import annotations

import numpy as np


def trim_orange_frame_rows(mask: np.ndarray, image_rgb: np.ndarray) -> dict[str, int | bool]:
    """Remove foreground on the two broad orange end rails, if both are clear.

    Only the top/bottom 10% are inspected. A rusty core cannot trigger this
    unless it forms a saturated, near-full-width horizontal bar in both ends.
    The input mask is modified in place to avoid a second 200 MP allocation.
    """

    height, width = mask.shape
    if image_rgb.shape[:2] != mask.shape or image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError("RGB image and mask must share a two-dimensional grid")
    sy = max(1, height // 2048)
    sx = max(1, width // 512)
    sample = image_rgb[::sy, ::sx, :3].astype(np.int16)
    red, green, blue = sample[:, :, 0], sample[:, :, 1], sample[:, :, 2]
    orange = (red > 125) & (red > 1.35 * green) & (green > 1.35 * blue) & (red - blue > 80)
    left, right = int(sample.shape[1] * 0.10), int(sample.shape[1] * 0.90)
    coverage = orange[:, left:right].mean(axis=1)
    band = max(1, int(len(coverage) * 0.10))

    def runs(indices: np.ndarray) -> list[tuple[int, int]]:
        if indices.size == 0:
            return []
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        return [(int(group[0]), int(group[-1]) + 1) for group in groups if len(group) >= 3]

    top = runs(np.flatnonzero(coverage[:band] > 0.35))
    bottom = runs(np.flatnonzero(coverage[-band:] > 0.35) + len(coverage) - band)
    if not top or not bottom:
        return {"applied": False, "top_end": 0, "bottom_start": height}
    top_rail = max(top, key=lambda item: item[1] - item[0])
    bottom_rail = max(bottom, key=lambda item: item[1] - item[0])
    top_end = min(height, top_rail[1] * sy)
    bottom_start = min(height, bottom_rail[0] * sy)
    if bottom_start <= top_end:
        return {"applied": False, "top_end": 0, "bottom_start": height}
    mask[:top_end] = False
    mask[bottom_start:] = False
    return {"applied": True, "top_end": top_end, "bottom_start": bottom_start}
