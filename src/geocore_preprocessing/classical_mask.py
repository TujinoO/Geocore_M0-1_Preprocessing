from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Small dependency-free connected component filter for fallback mode."""

    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny < 0 or nx < 0 or ny >= height or nx >= width or visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(pixels) > len(best_pixels):
                best_pixels = pixels
    out = np.zeros(mask.shape, dtype=np.uint8)
    for y, x in best_pixels:
        out[y, x] = 1
    return out


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32).copy()
    color = np.array([0, 190, 255], dtype=np.float32)
    selected = mask.astype(bool)
    out[selected] = out[selected] * 0.55 + color * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def run_classical_foreground_mask(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float | None = None,
    keep_largest_component: bool = False,
) -> dict[str, Any]:
    """Create a lightweight foreground mask when the trained M1-2 model is unavailable.

    This is a pipeline continuity fallback, not the preferred production model.
    """

    started = time.time()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image = _read_rgb(image_path)
    gray = image.astype(np.float32).mean(axis=2)
    non_black = np.max(image, axis=2) > 6
    dynamic_threshold = float(threshold * 255.0) if threshold is not None and threshold <= 1 else threshold
    if dynamic_threshold is None:
        valid = gray[non_black]
        dynamic_threshold = float(np.percentile(valid, 5)) if valid.size else 8.0
        dynamic_threshold = max(8.0, dynamic_threshold)
    mask = np.logical_and(non_black, gray >= dynamic_threshold).astype(np.uint8)
    if keep_largest_component:
        mask = _largest_component_mask(mask)

    mask_png = output / "mask.png"
    mask_tif = output / "mask.tif"
    probability_tif = output / "probability.tif"
    overlay_png = output / "overlay.png"
    contours_json = output / "contours.json"
    metadata_json = output / "metadata.json"

    _save_mask(mask, mask_png)
    _save_mask(mask, mask_tif)
    _save_mask(mask, probability_tif)
    Image.fromarray(_overlay(image, mask), mode="RGB").save(overlay_png)
    bbox = _bbox_from_mask(mask)
    contours = {
        "component_count": 1 if bbox else 0,
        "components": [{"component_id": 1, "bbox_xyxy": bbox, "area_px": int(mask.sum())}] if bbox else [],
    }
    metadata = {
        "module": "M1-2 foreground mask",
        "engine": "classical_fallback",
        "input_path": str(image_path),
        "threshold": dynamic_threshold,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "foreground_area_ratio": float(mask.mean()) if mask.size else 0.0,
        "component_count": int(contours["component_count"]),
        "warnings": ["classical fallback was used instead of the trained M1-2 model"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    contours_json.write_text(json.dumps(contours, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_files": {
            "mask_png": str(mask_png),
            "mask_tif": str(mask_tif),
            "probability_tif": str(probability_tif),
            "overlay_png": str(overlay_png),
            "contours_json": str(contours_json),
            "metadata_json": str(metadata_json),
        },
        "metrics": {
            "foreground_area_ratio": metadata["foreground_area_ratio"],
            "component_count": metadata["component_count"],
            "elapsed_seconds": metadata["elapsed_seconds"],
        },
        "warnings": metadata["warnings"],
    }
