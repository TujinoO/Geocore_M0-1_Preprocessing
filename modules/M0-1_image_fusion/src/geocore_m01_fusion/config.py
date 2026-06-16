"""Configuration objects for M0-1 fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FusionMode(str, Enum):
    """Supported M0-1 algorithm modes."""

    UPSAMPLE_ONLY = "upsample_only"
    FAST_PREVIEW = "fast_preview"
    CLASSICAL = "classical"
    DEEP_UNSUPERVISED = "deep_unsupervised"


@dataclass(slots=True)
class FusionConfig:
    """Runtime configuration for the fusion pipeline."""

    mode: FusionMode | str = FusionMode.CLASSICAL
    interpolation: str = "bilinear"
    output_dtype: str = "float32"
    chunk_size: tuple[int, int, int] = (512, 512, 32)
    rank: int = 12
    max_basis_pixels: int = 20000
    fast_detail_strength: float = 0.26
    classical_detail_strength: float = 0.12
    deep_detail_strength: float = 0.10
    spatial_detail_strength: float = 0.42
    spatial_detail_small_radius: int = 2
    spatial_detail_large_radius: int = 9
    deep_iterations: int = 4
    back_projection_iters: int = 2
    back_projection_weight: float = 0.60
    mask_background_value: float = 0.0
    random_seed: int = 7
    export_envi: bool = False
    write_previews: bool = True
    streaming: bool = False
    streaming_preview_max_size: int = 1024
    algorithm_version: str = "0.1.0"
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized_mode(self) -> FusionMode:
        """Return the mode as a validated enum."""

        if isinstance(self.mode, FusionMode):
            return self.mode
        try:
            return FusionMode(str(self.mode))
        except ValueError as exc:
            allowed = ", ".join(mode.value for mode in FusionMode)
            raise ValueError(f"Unknown fusion mode {self.mode!r}. Allowed: {allowed}") from exc


@dataclass(slots=True)
class EnviInput:
    """Path pair for an ENVI image."""

    hdr_path: Path
    dat_path: Path | None = None


@dataclass(slots=True)
class ProjectMetadata:
    """Optional geological project metadata recorded in manifest.json."""

    project_id: str | None = None
    borehole_id: str | None = None
    box_id: str | None = None
    depth_start_m: float | None = None
    depth_end_m: float | None = None
