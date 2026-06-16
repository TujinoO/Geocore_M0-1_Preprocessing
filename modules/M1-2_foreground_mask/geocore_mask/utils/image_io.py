from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    from skimage import exposure, img_as_ubyte, io
except Exception:
    exposure = None
    img_as_ubyte = None
    io = None


@dataclass
class ImageData:
    array: np.ndarray
    path: Path
    projection: Optional[str] = None
    geotransform: Optional[tuple[float, ...]] = None


def _ensure_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3 and array.shape[0] <= 16 and array.shape[0] < array.shape[-1]:
        return np.moveaxis(array, 0, -1)
    return array


def read_image(path: str | Path) -> ImageData:
    image_path = Path(path)
    projection = None
    geotransform = None

    try:
        from osgeo import gdal

        ds = gdal.Open(str(image_path))
        if ds is not None:
            projection = ds.GetProjection() or None
            gt = ds.GetGeoTransform(can_return_null=True)
            geotransform = tuple(gt) if gt else None
            array = ds.ReadAsArray()
            ds = None
            return ImageData(array=_ensure_hwc(np.asarray(array)), path=image_path, projection=projection, geotransform=geotransform)
    except Exception:
        pass

    if io is not None:
        array = io.imread(str(image_path))
    else:
        from PIL import Image

        with Image.open(image_path) as img:
            array = np.asarray(img)
    return ImageData(array=_ensure_hwc(np.asarray(array)), path=image_path)


def prepare_display_rgb(image: np.ndarray) -> np.ndarray:
    image = _ensure_hwc(image)
    if image.shape[2] == 1:
        rgb = np.repeat(image, 3, axis=2)
    else:
        rgb = image[:, :, :3]

    if rgb.dtype == np.uint8:
        return rgb

    rgb_float = rgb.astype(np.float32)
    out = np.zeros(rgb_float.shape, dtype=np.float32)
    for c in range(rgb_float.shape[2]):
        channel = rgb_float[:, :, c]
        p2, p98 = np.percentile(channel, (2, 98))
        if p98 > p2:
            if exposure is not None:
                out[:, :, c] = exposure.rescale_intensity(channel, in_range=(p2, p98), out_range=(0, 1))
            else:
                out[:, :, c] = np.clip((channel - p2) / (p98 - p2), 0, 1)
        else:
            out[:, :, c] = 0
    if img_as_ubyte is not None:
        return img_as_ubyte(np.clip(out, 0, 1))
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path: str | Path) -> str:
    out = (mask.astype(np.uint8) * 255)
    if io is not None:
        io.imsave(str(path), out, check_contrast=False)
    else:
        from PIL import Image

        Image.fromarray(out).save(path)
    return str(path)


def save_overlay_png(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.45) -> str:
    rgb = prepare_display_rgb(image).astype(np.float32)
    color = np.array([0, 190, 255], dtype=np.float32)
    overlay = rgb.copy()
    overlay[mask.astype(bool)] = (1 - alpha) * overlay[mask.astype(bool)] + alpha * color
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    if io is not None:
        io.imsave(str(path), out, check_contrast=False)
    else:
        from PIL import Image

        Image.fromarray(out).save(path)
    return str(path)


def save_tiff_like(
    array: np.ndarray,
    path: str | Path,
    *,
    reference: Optional[ImageData] = None,
    dtype: Any = np.uint8,
) -> str:
    out = np.asarray(array).astype(dtype)
    try:
        from osgeo import gdal

        gdal_dtype = gdal.GDT_Byte if out.dtype == np.uint8 else gdal.GDT_Float32
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), out.shape[1], out.shape[0], 1, gdal_dtype, options=["COMPRESS=DEFLATE", "TILED=YES"])
        if reference and reference.geotransform:
            ds.SetGeoTransform(reference.geotransform)
        if reference and reference.projection:
            ds.SetProjection(reference.projection)
        ds.GetRasterBand(1).WriteArray(out)
        ds.FlushCache()
        ds = None
    except Exception:
        if io is not None:
            io.imsave(str(path), out, check_contrast=False)
        else:
            from PIL import Image

            Image.fromarray(out).save(path)
    return str(path)
