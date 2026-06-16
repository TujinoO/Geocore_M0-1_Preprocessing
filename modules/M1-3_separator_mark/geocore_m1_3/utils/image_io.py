from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def read_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.uint8) > threshold


def save_rgb(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(target)


def save_rgba(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGBA").save(target)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(target)


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 180, 220), alpha: float = 0.45) -> np.ndarray:
    overlay = image_rgb.astype(np.float32).copy()
    color_array = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_array * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_lane_debug(path: str | Path, image_rgb: np.ndarray, lanes: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    colors = ["red", "yellow", "lime", "cyan", "magenta", "orange", "white"]
    for idx, lane in enumerate(lanes):
        x0, y0, x1, y1 = lane["bbox"]
        color = colors[idx % len(colors)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 4, max(0, y0 + 4)), f"L{lane['lane_index']}", fill=color)
    canvas.save(target)
