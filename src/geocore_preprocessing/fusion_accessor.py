from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageGrid:
    width: int
    height: int
    name: str


class FusionResultAccessor:
    """Read the M0-1 output contract without hard-coding concrete file paths."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @property
    def grid(self) -> ImageGrid:
        grid = self.manifest.get("grid", {})
        return ImageGrid(
            width=int(grid.get("width", self.manifest.get("cube", {}).get("shape", [0, 0])[1])),
            height=int(grid.get("height", self.manifest.get("cube", {}).get("shape", [0, 0])[0])),
            name=str(grid.get("reference", "rgb")),
        )

    def resolve(self, path_value: str | Path | None) -> Path | None:
        if not path_value:
            return None
        path = Path(path_value)
        return path if path.is_absolute() else self.root / path

    def preview_path(self, preferred: Iterable[str] = ("preview_rgb", "rgb", "preview_falsecolor_full")) -> Path:
        previews = self.manifest.get("previews", {})
        for key in preferred:
            path = self.resolve(previews.get(key))
            if path and path.exists():
                return path
        rgb_reference = self.manifest.get("rgb_reference", {})
        path = self.resolve(rgb_reference.get("preview"))
        if path and path.exists():
            return path
        raise FileNotFoundError(f"No usable RGB preview was found in manifest: {self.manifest_path}")

    @property
    def cube_path(self) -> Path:
        path = self.resolve(self.manifest.get("cube", {}).get("path"))
        if path is None:
            raise KeyError("manifest.cube.path is required")
        return path

    @property
    def band_metadata_path(self) -> Path:
        path = self.resolve(self.manifest.get("metadata", {}).get("band_metadata"))
        if path is None:
            raise KeyError("manifest.metadata.band_metadata is required")
        return path

    @property
    def registration_model_path(self) -> Path | None:
        return self.resolve(self.manifest.get("metadata", {}).get("registration_model"))

    @property
    def quality_report_path(self) -> Path | None:
        return self.resolve(self.manifest.get("metadata", {}).get("quality_report"))

    def preview_grid(self, preview_path: str | Path | None = None) -> ImageGrid:
        path = Path(preview_path) if preview_path else self.preview_path()
        with Image.open(path) as image:
            return ImageGrid(width=int(image.width), height=int(image.height), name="m1_working_preview")

    def scale_from_preview_to_reference(self, preview_path: str | Path | None = None) -> tuple[float, float]:
        preview = self.preview_grid(preview_path)
        grid = self.grid
        return (grid.width / max(1, preview.width), grid.height / max(1, preview.height))

    def bbox_preview_to_reference(self, bbox_xyxy: list[int] | tuple[int, int, int, int], preview_path: str | Path | None = None) -> list[int]:
        sx, sy = self.scale_from_preview_to_reference(preview_path)
        x0, y0, x1, y1 = bbox_xyxy
        grid = self.grid
        return [
            int(np.clip(round(x0 * sx), 0, grid.width)),
            int(np.clip(round(y0 * sy), 0, grid.height)),
            int(np.clip(round(x1 * sx), 0, grid.width)),
            int(np.clip(round(y1 * sy), 0, grid.height)),
        ]

    def band_metadata(self) -> list[dict[str, str]]:
        with self.band_metadata_path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def selected_band_indices(self, purpose: str = "mineral") -> list[int]:
        flag_name = "recommended_for_mineral" if purpose == "mineral" else "recommended_for_visualization"
        selected: list[int] = []
        for row in self.band_metadata():
            bad = str(row.get("is_bad_band", "")).strip().lower() in {"1", "true", "yes", "y"}
            recommended = str(row.get(flag_name, "true")).strip().lower() not in {"0", "false", "no", "n"}
            if bad or not recommended:
                continue
            index = row.get("fused_band_index") or row.get("band_index") or row.get("index")
            if index is not None and str(index).strip() != "":
                selected.append(int(float(index)))
        return selected

    def cube_ref(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "cube_path": str(self.cube_path),
            "array": self.manifest.get("cube", {}).get("array", "reflectance"),
            "band_metadata": str(self.band_metadata_path),
            "registration_model": str(self.registration_model_path) if self.registration_model_path else None,
            "axis_order": self.manifest.get("grid", {}).get("axis_order", "y,x,band"),
            "grid": self.manifest.get("grid", {}),
        }

    def read_zarr_window(
        self,
        bbox_xyxy: tuple[int, int, int, int] | list[int],
        bands: list[int] | None = None,
    ) -> np.ndarray:
        """Read a y,x,band window from the minimal uncompressed Zarr v2 writer used by M0-1."""

        root = self.cube_path
        meta = json.loads((root / ".zarray").read_text(encoding="utf-8"))
        shape = tuple(int(x) for x in meta["shape"])
        chunks = tuple(int(x) for x in meta["chunks"])
        dtype = np.dtype(meta["dtype"])
        fill = meta.get("fill_value", 0.0)
        x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
        x0, x1 = sorted((max(0, x0), min(shape[1], x1)))
        y0, y1 = sorted((max(0, y0), min(shape[0], y1)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Empty cube window: {bbox_xyxy}")
        band_indices = bands if bands is not None else list(range(shape[2]))
        out = np.full((y1 - y0, x1 - x0, len(band_indices)), fill, dtype=dtype)

        cy, cx, cb = chunks
        band_to_out = {int(band): i for i, band in enumerate(band_indices)}
        for yy in range((y0 // cy) * cy, y1, cy):
            for xx in range((x0 // cx) * cx, x1, cx):
                for bb in range(0, shape[2], cb):
                    chunk_bands = range(bb, min(bb + cb, shape[2]))
                    wanted = [band for band in chunk_bands if band in band_to_out]
                    if not wanted:
                        continue
                    chunk_path = root / f"{yy // cy}.{xx // cx}.{bb // cb}"
                    if not chunk_path.exists():
                        continue
                    chunk = np.fromfile(chunk_path, dtype=dtype).reshape(chunks)
                    yy1, xx1 = min(yy + cy, shape[0]), min(xx + cx, shape[1])
                    iy0, iy1 = max(y0, yy), min(y1, yy1)
                    ix0, ix1 = max(x0, xx), min(x1, xx1)
                    for band in wanted:
                        out[:, :, band_to_out[band]][iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = chunk[
                            iy0 - yy : iy1 - yy,
                            ix0 - xx : ix1 - xx,
                            band - bb,
                        ]
        return out
