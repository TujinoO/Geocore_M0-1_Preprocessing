from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ENVI_DTYPE_MAP: dict[int, Any] = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def parse_envi_header(hdr_path: str | Path) -> dict[str, Any]:
    """Parse a small ENVI header into a lowercase-key dictionary."""

    text = Path(hdr_path).read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    meta: dict[str, Any] = {}
    current_key: str | None = None
    current_value: list[str] = []

    for line in lines:
        if line.upper() == "ENVI":
            continue
        if current_key is not None:
            current_value.append(line)
            if "}" in line:
                meta[current_key] = " ".join(current_value).strip()
                current_key = None
                current_value = []
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        key = key.lower()
        if value.startswith("{") and "}" not in value:
            current_key = key
            current_value = [value]
        else:
            meta[key] = value

    for key in ("samples", "lines", "bands", "header offset", "data type", "byte order"):
        if key in meta:
            match = re.search(r"-?\d+", str(meta[key]))
            if match:
                meta[key] = int(match.group(0))
    if "interleave" in meta:
        meta["interleave"] = str(meta["interleave"]).lower()
    return meta


def read_envi_image(dat_path: str | Path, hdr_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an ENVI raster as an HWC NumPy array view when possible."""

    meta = parse_envi_header(hdr_path)
    try:
        lines = int(meta["lines"])
        samples = int(meta["samples"])
        bands = int(meta["bands"])
        data_type = int(meta["data type"])
    except KeyError as exc:
        raise ValueError(f"Missing required ENVI header key: {exc}") from exc

    dtype = np.dtype(ENVI_DTYPE_MAP.get(data_type, np.uint8))
    if int(meta.get("byte order", 0)) == 1:
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    offset = int(meta.get("header offset", 0))
    interleave = str(meta.get("interleave", "bil")).lower()
    path = Path(dat_path)

    if interleave == "bil":
        raw = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(lines, bands, samples))
        image = np.transpose(raw, (0, 2, 1))
    elif interleave == "bsq":
        raw = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(bands, lines, samples))
        image = np.transpose(raw, (1, 2, 0))
    elif interleave == "bip":
        image = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(lines, samples, bands))
    else:
        raise ValueError(f"Unsupported ENVI interleave: {interleave}")

    if bands == 3 and "default bands" in meta:
        numbers = [int(value) for value in re.findall(r"\d+", str(meta["default bands"]))]
        if len(numbers) != 3:
            raise ValueError(f"Expected three RGB default bands in {hdr_path}: {numbers}")
        if set(numbers) == {0, 1, 2}:
            order = numbers
        elif set(numbers) == {1, 2, 3}:
            order = [value - 1 for value in numbers]
        else:
            raise ValueError(f"Invalid RGB default bands in {hdr_path}: {numbers}")
        if order == [2, 1, 0]:
            image = image[:, :, ::-1]
        elif order != [0, 1, 2]:
            raise ValueError(f"RGB order {numbers} requires a copy of the full ENVI scan; convert it before M1-1")
        meta["rgb_bands_zero_based"] = order
    elif bands == 3:
        meta["rgb_bands_zero_based"] = [0, 1, 2]
        meta["rgb_order_assumed"] = True

    meta["shape_hwc"] = tuple(int(v) for v in image.shape)
    return image, meta


def read_input_image(input_path: str | Path, hdr_path: str | Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Read ENVI or common RGB image input."""

    path = Path(input_path)
    if hdr_path is not None or path.suffix.lower() == ".dat":
        if hdr_path is None:
            candidate = path.with_suffix(".hdr")
            if not candidate.exists():
                raise ValueError("ENVI .dat input requires a matching .hdr path.")
            hdr_path = candidate
        return read_envi_image(path, hdr_path)

    image = np.asarray(Image.open(path).convert("RGB"))
    return image, {"shape_hwc": tuple(int(v) for v in image.shape), "source_type": "image"}


def save_rgb_image(path: str | Path, image: np.ndarray, quality: int = 92) -> None:
    """Save a 2D or RGB image using Pillow."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        mode = "L"
        out = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        mode = "RGB"
        out = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        mode = "L"
        out = arr[:, :, 0]
    else:
        raise ValueError(f"Cannot save image with shape {arr.shape}")

    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, mode=mode)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        img.save(path, quality=quality)
    else:
        img.save(path)
