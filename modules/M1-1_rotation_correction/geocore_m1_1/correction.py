from __future__ import annotations

import numpy as np

from .config import M11Config
from .image_ops import bbox_from_mask, pad_bbox, rotate_nearest


def correct_core_box(
    image: np.ndarray,
    bbox_xyxy_raw: tuple[int, int, int, int],
    angle_deg: float,
    config: M11Config,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], list[list[float]], tuple[int, int, int, int]]:
    """Crop, rotate with nearest neighbor, and tightly crop a single core box."""

    raw_h, raw_w = image.shape[:2]
    padded_bbox = pad_bbox(bbox_xyxy_raw, raw_w, raw_h, config.raw_bbox_margin_px)
    px0, py0, px1, py1 = padded_bbox
    x0, y0, x1, y1 = bbox_xyxy_raw
    crop = np.asarray(image[py0 : py1 + 1, px0 : px1 + 1])
    inner = (
        max(0, x0 - px0),
        max(0, y0 - py0),
        min(px1 - px0, x1 - px0),
        min(py1 - py0, y1 - py0),
    )
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    ix0, iy0, ix1, iy1 = inner
    mask[iy0 : iy1 + 1, ix0 : ix1 + 1] = 255

    rotated_image, matrix = rotate_nearest(crop, angle_deg, fill_value=config.fill_value)
    rotated_mask, _ = rotate_nearest(mask, angle_deg, fill_value=0)
    bbox = bbox_from_mask(rotated_mask > 0)
    if bbox is None:
        corrected_bbox = (0, 0, rotated_image.shape[1] - 1, rotated_image.shape[0] - 1)
    else:
        corrected_bbox = pad_bbox(bbox, rotated_image.shape[1], rotated_image.shape[0], config.crop_margin_px)

    cx0, cy0, cx1, cy1 = corrected_bbox
    corrected_image = rotated_image[cy0 : cy1 + 1, cx0 : cx1 + 1]
    corrected_mask = rotated_mask[cy0 : cy1 + 1, cx0 : cx1 + 1]
    return corrected_image, corrected_mask, corrected_bbox, matrix, inner

