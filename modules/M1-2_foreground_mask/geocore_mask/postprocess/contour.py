from __future__ import annotations

from typing import Any, Dict

import numpy as np
from skimage import measure


def mask_to_components(mask: np.ndarray, max_contour_points: int = 800) -> Dict[str, Any]:
    labels = measure.label(mask.astype(bool))
    components = []
    for prop in measure.regionprops(labels):
        component_mask = labels == prop.label
        contours = measure.find_contours(component_mask.astype(np.uint8), 0.5)
        contour = max(contours, key=len) if contours else np.empty((0, 2))
        if len(contour) > max_contour_points:
            step = max(1, len(contour) // max_contour_points)
            contour = contour[::step]
        min_row, min_col, max_row, max_col = prop.bbox
        components.append(
            {
                "component_id": int(prop.label),
                "area": int(prop.area),
                "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "centroid": [float(prop.centroid[1]), float(prop.centroid[0])],
                "contour": [[float(x), float(y)] for y, x in contour],
            }
        )

    return {
        "image_width": int(mask.shape[1]),
        "image_height": int(mask.shape[0]),
        "component_count": len(components),
        "components": components,
    }
